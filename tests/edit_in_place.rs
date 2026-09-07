//! Tests for in-place editing via `File::open_rw` (issue #32, Group C):
//! add, delete, and copy datasets and groups at any path.

use hdf5_pure::{
    AttrValue, CharacterSet, CompoundTypeBuilder, DType, Datatype, Error, File, FileBuilder,
    FormatError, Object, ReferenceType, ScaleOffset, StringPadding,
};

use temp::temp_path;
use test_util::temp;

use heap::has_fractal_heap;
use test_util::heap;

/// Write a starter file with one dataset, returning its path.
fn write_starter(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.create_dataset("original")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0]);
    b.write(path).unwrap();
}

#[test]
fn add_dataset_preserves_original_and_adds_new() {
    let path = temp_path("hdf5_pure_edit_add_one.h5");
    write_starter(&path);
    let size_before = std::fs::metadata(&path).unwrap().len();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("added", |b| {
                b.with_i32_data(&[10, 20, 30]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    // Only grew; existing bytes were not rewritten.
    let size_after = std::fs::metadata(&path).unwrap().len();
    assert!(size_after > size_before);

    let file = File::open(&path).unwrap();
    // Original dataset still intact.
    let orig = file.dataset("original").unwrap();
    assert_eq!(orig.dtype().unwrap(), DType::F64);
    assert_eq!(orig.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    // New dataset present and correct.
    let added = file.dataset("added").unwrap();
    assert_eq!(added.dtype().unwrap(), DType::I32);
    assert_eq!(added.read_i32().unwrap(), vec![10, 20, 30]);

    // Root group lists exactly the two datasets.
    let mut names = file.root().datasets().unwrap();
    names.sort();
    assert_eq!(names, vec!["added".to_string(), "original".to_string()]);
}

#[test]
fn add_multiple_datasets_in_one_commit() {
    let path = temp_path("hdf5_pure_edit_add_many.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("a", |b| {
                b.with_f64_data(&[1.5, 2.5]);
            })
            .unwrap();
        session
            .root()
            .create_dataset("b", |b| {
                b.with_i32_data(&[7, 8, 9]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("a").unwrap().read_f64().unwrap(),
        vec![1.5, 2.5]
    );
    assert_eq!(
        file.dataset("b").unwrap().read_i32().unwrap(),
        vec![7, 8, 9]
    );
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn successive_commits_accumulate() {
    let path = temp_path("hdf5_pure_edit_successive.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("first", |b| {
                b.with_i32_data(&[1]);
            })
            .unwrap();
        session.commit().unwrap();
        session
            .root()
            .create_dataset("second", |b| {
                b.with_i32_data(&[2]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("first").unwrap().read_i32().unwrap(), vec![1]);
    assert_eq!(file.dataset("second").unwrap().read_i32().unwrap(), vec![2]);
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn add_dataset_with_multidim_shape() {
    let path = temp_path("hdf5_pure_edit_2d.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("matrix", |b| {
                b.with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                    .with_shape(&[2, 3]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let m = file.dataset("matrix").unwrap();
    assert_eq!(m.shape().unwrap(), vec![2, 3]);
    assert_eq!(m.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn commit_without_staged_datasets_is_noop() {
    let path = temp_path("hdf5_pure_edit_noop.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session.commit().unwrap();
    }

    let after = std::fs::read(&path).unwrap();
    assert_eq!(before, after, "empty commit must not modify the file");
}

#[test]
fn create_group_at_root() {
    let path = temp_path("hdf5_pure_edit_group.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session.root().create_group("results").unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.root().groups().unwrap(), vec!["results".to_string()]);
    // The new group is empty and openable.
    assert!(
        file.group("results")
            .unwrap()
            .datasets()
            .unwrap()
            .is_empty()
    );
    // Original dataset intact.
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn add_dataset_into_new_nested_group() {
    let path = temp_path("hdf5_pure_edit_nested.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session.root().create_group("measurements").unwrap();
        session.root().create_group("measurements/run1").unwrap();
        session
            .root()
            .create_dataset("measurements/run1/signal", |b| {
                b.with_f64_data(&[10.0, 11.0, 12.0]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("measurements/run1/signal").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![10.0, 11.0, 12.0]);
    // Ancestors and the original survive.
    assert_eq!(
        file.group("measurements").unwrap().groups().unwrap(),
        vec!["run1".to_string()]
    );
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn add_into_existing_group_across_commits() {
    let path = temp_path("hdf5_pure_edit_existing_group.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session.root().create_group("g").unwrap();
        session
            .root()
            .create_dataset("g/a", |b| {
                b.with_i32_data(&[1, 2]);
            })
            .unwrap();
        session.commit().unwrap();
        // Second commit adds into the now-existing group g.
        session
            .root()
            .create_dataset("g/b", |b| {
                b.with_i32_data(&[3, 4]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("g/a").unwrap().read_i32().unwrap(), vec![1, 2]);
    assert_eq!(file.dataset("g/b").unwrap().read_i32().unwrap(), vec![3, 4]);
    let mut names = file.group("g").unwrap().datasets().unwrap();
    names.sort();
    assert_eq!(names, vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn add_into_two_sibling_groups_one_commit() {
    let path = temp_path("hdf5_pure_edit_siblings.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session.root().create_group("x").unwrap();
        session.root().create_group("y").unwrap();
        session
            .root()
            .create_dataset("x/d", |b| {
                b.with_i32_data(&[1]);
            })
            .unwrap();
        session
            .root()
            .create_dataset("y/d", |b| {
                b.with_i32_data(&[2]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("x/d").unwrap().read_i32().unwrap(), vec![1]);
    assert_eq!(file.dataset("y/d").unwrap().read_i32().unwrap(), vec![2]);
}

#[test]
fn dataset_into_missing_group_is_rejected() {
    let path = temp_path("hdf5_pure_edit_missing_group.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("nope/d", |b| {
                b.with_i32_data(&[1]);
            })
            .unwrap();
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("does not exist"), "got: {err}");
    }
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn duplicate_name_is_rejected_without_writing() {
    let path = temp_path("hdf5_pure_edit_dup.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    // Collide with the existing "original" dataset. Refused where it is staged,
    // since `create_dataset` returns a handle onto what it stages and there is
    // no new object for one to address.
    {
        let session = File::open_rw(&path).unwrap();
        let err = session
            .root()
            .create_dataset("original", |b| {
                b.with_i32_data(&[1, 2]);
            })
            .unwrap_err();
        assert!(err.to_string().contains("already exists"), "got: {err}");
    }
    assert_eq!(std::fs::read(&path).unwrap(), before);

    // Collide between two datasets staged in the same commit. Neither shadows an
    // object in the file, but the by-path index keeps the first record, so the
    // handle the second call returns would answer for the first creation until
    // the commit refused the batch. Refused where it is staged, for the same
    // reason the file collision above is.
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("dup", |b| {
                b.with_i32_data(&[1]);
            })
            .unwrap();
        let err = session
            .root()
            .create_dataset("dup", |b| {
                b.with_f64_data(&[2.0, 3.0]).with_shape(&[2]);
            })
            .unwrap_err();
        assert!(err.to_string().contains("already stages"), "got: {err}");
    }
    assert_eq!(std::fs::read(&path).unwrap(), before);

    // A group creation collides with a staged dataset, and a dataset creation
    // with a staged group: one path, two objects, and a handle onto either
    // reports the kind the index found first.
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("mixed", |b| {
                b.with_i32_data(&[1]);
            })
            .unwrap();
        let err = session.root().create_group("mixed").unwrap_err();
        assert!(err.to_string().contains("already stages"), "got: {err}");
    }
    {
        let session = File::open_rw(&path).unwrap();
        session.root().create_group("mixed").unwrap();
        let err = session
            .root()
            .create_dataset("mixed", |b| {
                b.with_i32_data(&[1]);
            })
            .unwrap_err();
        assert!(err.to_string().contains("already stages"), "got: {err}");
    }
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

/// The refusal above leaves the session holding exactly what it held before the
/// second call: the first creation still commits, and its handle still addresses
/// it. A rejected duplicate must not withdraw the staging it collided with.
#[test]
fn refused_duplicate_leaves_the_first_creation_standing() {
    let path = temp_path("hdf5_pure_edit_dup_first_stands.h5");
    write_starter(&path);

    let session = File::open_rw(&path).unwrap();
    let first = session
        .root()
        .create_dataset("dup", |b| {
            b.with_i32_data(&[1, 2, 3]);
        })
        .unwrap();
    let err = session
        .root()
        .create_dataset("dup", |b| {
            b.with_f64_data(&[4.0, 5.0]).with_shape(&[2]);
        })
        .unwrap_err();
    assert!(err.to_string().contains("already stages"), "got: {err}");

    // The surviving handle answers for what it staged, not for the refused call.
    assert_eq!(first.shape().unwrap(), vec![3]);
    assert_eq!(first.dtype().unwrap(), DType::I32);
    session.commit().unwrap();
    assert_eq!(first.read_i32().unwrap(), vec![1, 2, 3]);
    // Windows refuses a second open while the session still holds its lock.
    drop(first);
    drop(session);

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("dup").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
}

/// Deleting a staged creation withdraws it, which is what makes the path free
/// again: stage, withdraw, stage another, and the second one is the one that
/// commits.
#[test]
fn withdrawing_a_staged_creation_frees_its_name_again() {
    let path = temp_path("hdf5_pure_edit_dup_withdraw.h5");
    write_starter(&path);

    let session = File::open_rw(&path).unwrap();
    session
        .root()
        .create_dataset("swap", |b| {
            b.with_i32_data(&[1, 2, 3]);
        })
        .unwrap();
    session.root().delete("swap").unwrap();
    let second = session
        .root()
        .create_dataset("swap", |b| {
            b.with_f64_data(&[4.0, 5.0]).with_shape(&[2]);
        })
        .unwrap();
    assert_eq!(second.shape().unwrap(), vec![2]);
    session.commit().unwrap();
    // Windows refuses a second open while the session still holds its lock.
    drop(second);
    drop(session);

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("swap").unwrap().read_f64().unwrap(),
        vec![4.0, 5.0]
    );
}

/// Two group creations at one path stay allowed: they name one node, so a
/// handle onto either addresses the group the commit builds. Re-staging is how
/// attributes and children are added to a group already staged.
#[test]
fn restaging_a_group_addresses_the_same_group() {
    let path = temp_path("hdf5_pure_edit_dup_group.h5");
    write_starter(&path);

    let session = File::open_rw(&path).unwrap();
    let first = session.root().create_group("g").unwrap();
    first.set_attr("from", AttrValue::I64(1)).unwrap();
    let again = session.root().create_group("g").unwrap();
    again.set_attr("also", AttrValue::I64(2)).unwrap();
    again
        .create_dataset("inner", |b| {
            b.with_i32_data(&[7]);
        })
        .unwrap();
    session.commit().unwrap();
    // Windows refuses a second open while the session still holds its lock.
    drop(again);
    drop(first);
    drop(session);

    let file = File::open(&path).unwrap();
    let attrs = file.group("g").unwrap().attrs().unwrap();
    assert_eq!(attrs.get("from"), Some(&AttrValue::I64(1)));
    assert_eq!(attrs.get("also"), Some(&AttrValue::I64(2)));
    assert_eq!(
        file.dataset("g/inner").unwrap().read_i32().unwrap(),
        vec![7]
    );
}

#[test]
fn delete_dataset_from_root() {
    let path = temp_path("hdf5_pure_edit_del_root.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.create_dataset("remove").with_i32_data(&[9, 9]);
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("remove").unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert!(file.dataset("remove").is_err());
}

#[test]
fn delete_nested_group_subtree() {
    let path = temp_path("hdf5_pure_edit_del_nested.h5");
    write_starter(&path);
    {
        let session = File::open_rw(&path).unwrap();
        session.root().create_group("grp").unwrap();
        session
            .root()
            .create_dataset("grp/inner", |b| {
                b.with_i32_data(&[5, 6]);
            })
            .unwrap();
        session
            .root()
            .create_dataset("sibling", |b| {
                b.with_i32_data(&[7]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    // Delete the whole group "grp" (its subtree becomes unreachable).
    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("grp").unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert!(file.group("grp").is_err());
    assert!(file.dataset("grp/inner").is_err());
    // Siblings and original survive.
    assert_eq!(
        file.dataset("sibling").unwrap().read_i32().unwrap(),
        vec![7]
    );
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    let mut roots = file.root().datasets().unwrap();
    roots.sort();
    assert_eq!(roots, vec!["original".to_string(), "sibling".to_string()]);
    assert!(file.root().groups().unwrap().is_empty());
}

#[test]
fn delete_one_of_nested_then_keep_group() {
    let path = temp_path("hdf5_pure_edit_del_one.h5");
    write_starter(&path);
    {
        let session = File::open_rw(&path).unwrap();
        session.root().create_group("g").unwrap();
        session
            .root()
            .create_dataset("g/a", |b| {
                b.with_i32_data(&[1]);
            })
            .unwrap();
        session
            .root()
            .create_dataset("g/b", |b| {
                b.with_i32_data(&[2]);
            })
            .unwrap();
        session.commit().unwrap();
    }
    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("g/a").unwrap(); // remove one member, keep the group and g/b
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert!(file.dataset("g/a").is_err());
    assert_eq!(file.dataset("g/b").unwrap().read_i32().unwrap(), vec![2]);
    assert_eq!(
        file.group("g").unwrap().datasets().unwrap(),
        vec!["b".to_string()]
    );
}

#[test]
fn add_and_delete_in_one_commit() {
    let path = temp_path("hdf5_pure_edit_add_del.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("old").with_i32_data(&[1]);
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("new", |b| {
                b.with_i32_data(&[2]);
            })
            .unwrap();
        session.root().delete("old").unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), vec!["new".to_string()]);
    assert_eq!(file.dataset("new").unwrap().read_i32().unwrap(), vec![2]);
    assert!(file.dataset("old").is_err());
}

#[test]
fn delete_missing_or_overlapping_is_rejected() {
    let path = temp_path("hdf5_pure_edit_del_reject.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    // Nothing to delete.
    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("ghost").unwrap();
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("nothing to delete"), "got: {err}");
    }
    assert_eq!(std::fs::read(&path).unwrap(), before);

    // Delete /g while adding under it in the same commit → overlap rejected.
    {
        let session = File::open_rw(&path).unwrap();
        session.root().create_group("g").unwrap();
        session.commit().unwrap();
    }
    let mid = std::fs::read(&path).unwrap();
    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("g").unwrap();
        session
            .root()
            .create_dataset("g/x", |b| {
                b.with_i32_data(&[1]);
            })
            .unwrap();
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("overlaps"), "got: {err}");
    }
    assert_eq!(std::fs::read(&path).unwrap(), mid);
}

#[test]
fn copy_dataset_to_new_name() {
    let path = temp_path("hdf5_pure_edit_copy_ds.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("src").with_f64_data(&[1.5, 2.5, 3.5]);
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session.copy("src", "dup").unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    // Original and copy both present and identical.
    assert_eq!(
        file.dataset("src").unwrap().read_f64().unwrap(),
        vec![1.5, 2.5, 3.5]
    );
    assert_eq!(
        file.dataset("dup").unwrap().read_f64().unwrap(),
        vec![1.5, 2.5, 3.5]
    );
    assert_eq!(file.dataset("dup").unwrap().dtype().unwrap(), DType::F64);
}

#[test]
fn copy_group_subtree() {
    let path = temp_path("hdf5_pure_edit_copy_grp.h5");
    write_starter(&path);
    {
        let session = File::open_rw(&path).unwrap();
        session.root().create_group("template").unwrap();
        session.root().create_group("template/inner").unwrap();
        session
            .root()
            .create_dataset("template/a", |b| {
                b.with_i32_data(&[1, 2]);
            })
            .unwrap();
        session
            .root()
            .create_dataset("template/inner/b", |b| {
                b.with_f64_data(&[9.0]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    // Copy the whole subtree under a new name.
    {
        let session = File::open_rw(&path).unwrap();
        session.copy("template", "run1").unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    // Deep structure duplicated.
    assert_eq!(
        file.dataset("run1/a").unwrap().read_i32().unwrap(),
        vec![1, 2]
    );
    assert_eq!(
        file.dataset("run1/inner/b").unwrap().read_f64().unwrap(),
        vec![9.0]
    );
    assert_eq!(
        file.group("run1").unwrap().groups().unwrap(),
        vec!["inner".to_string()]
    );
    // Original subtree untouched.
    assert_eq!(
        file.dataset("template/a").unwrap().read_i32().unwrap(),
        vec![1, 2]
    );
}

#[test]
fn copy_into_subgroup() {
    let path = temp_path("hdf5_pure_edit_copy_into.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("payload").with_i32_data(&[7, 8, 9]);
    b.write(&path).unwrap();
    {
        let session = File::open_rw(&path).unwrap();
        session.root().create_group("dest").unwrap();
        session.copy("payload", "dest/payload_copy").unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("dest/payload_copy")
            .unwrap()
            .read_i32()
            .unwrap(),
        vec![7, 8, 9]
    );
    assert_eq!(
        file.dataset("payload").unwrap().read_i32().unwrap(),
        vec![7, 8, 9]
    );
}

#[test]
fn copy_rejects_missing_source_and_cycle() {
    let path = temp_path("hdf5_pure_edit_copy_reject.h5");
    write_starter(&path);
    {
        let session = File::open_rw(&path).unwrap();
        session.root().create_group("g").unwrap();
        session.commit().unwrap();
    }
    let before = std::fs::read(&path).unwrap();

    // Missing source.
    {
        let session = File::open_rw(&path).unwrap();
        session.copy("ghost", "x").unwrap();
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string().contains("source does not exist"),
            "got: {err}"
        );
    }
    assert_eq!(std::fs::read(&path).unwrap(), before);

    // Copy a group into its own subtree.
    {
        let session = File::open_rw(&path).unwrap();
        session.copy("g", "g/inside").unwrap();
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("itself"), "got: {err}");
    }
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn add_dataset_with_attributes() {
    let path = temp_path("hdf5_pure_edit_add_attrs.h5");
    write_starter(&path);
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("measured", |ds| {
                ds.with_f64_data(&[1.0, 2.0]);
                ds.set_attr("count", AttrValue::I64(2));
                ds.set_attr("unit", AttrValue::String("m/s".into()));
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("measured").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0]);
    let attrs = ds.attrs().unwrap();
    assert_eq!(attrs.get("count"), Some(&AttrValue::I64(2)));
    assert_eq!(attrs.get("unit"), Some(&AttrValue::String("m/s".into())));
}

#[test]
fn create_group_with_attributes() {
    let path = temp_path("hdf5_pure_edit_group_attrs.h5");
    write_starter(&path);
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_group_with("run", |g| {
                g.set_attr("kind", AttrValue::AsciiString("trial".into()));
                g.set_attr("count", AttrValue::I64(2));
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let attrs = file.group("run").unwrap().attrs().unwrap();
    assert_eq!(
        attrs.get("kind"),
        Some(&AttrValue::AsciiString("trial".into()))
    );
    assert_eq!(attrs.get("count"), Some(&AttrValue::I64(2)));
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn edit_existing_group_attributes() {
    let path = temp_path("hdf5_pure_edit_existing_group_attrs.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("grp");
    g.set_attr("status", AttrValue::String("old".into()));
    g.set_attr("drop", AttrValue::I64(1));
    g.create_dataset("data").with_i32_data(&[5, 6]);
    b.add_group(g.finish());
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .group("grp")
            .unwrap()
            .set_attr("status", AttrValue::String("new".into()))
            .unwrap();
        session
            .group("grp")
            .unwrap()
            .set_attr("added", AttrValue::F64(3.5))
            .unwrap();
        session.group("grp").unwrap().remove_attr("drop").unwrap();
        session
            .root()
            .set_attr("root_tag", AttrValue::U64(9))
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let grp_attrs = file.group("grp").unwrap().attrs().unwrap();
    assert_eq!(
        grp_attrs.get("status"),
        Some(&AttrValue::String("new".into()))
    );
    assert_eq!(grp_attrs.get("added"), Some(&AttrValue::F64(3.5)));
    assert!(!grp_attrs.contains_key("drop"));
    assert_eq!(
        file.group("grp").unwrap().datasets().unwrap(),
        vec!["data".to_string()]
    );
    assert_eq!(
        file.root().attrs().unwrap().get("root_tag"),
        Some(&AttrValue::U64(9))
    );
}

#[test]
fn group_attribute_edit_uses_final_compact_count() {
    let path = temp_path("hdf5_pure_edit_group_attr_final_count.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("grp");
    for i in 0..8 {
        g.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    b.add_group(g.finish());
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .group("grp")
            .unwrap()
            .set_attr("new", AttrValue::I64(99))
            .unwrap();
        session.group("grp").unwrap().remove_attr("a0").unwrap();
        session.commit().unwrap();
    }

    let attrs = File::open(&path)
        .unwrap()
        .group("grp")
        .unwrap()
        .attrs()
        .unwrap();
    assert_eq!(attrs.len(), 8);
    assert!(!attrs.contains_key("a0"));
    assert_eq!(attrs.get("new"), Some(&AttrValue::I64(99)));
}

#[test]
fn remove_missing_group_attribute_is_rejected_without_writing() {
    let path = temp_path("hdf5_pure_edit_missing_group_attr.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("grp");
    g.set_attr("present", AttrValue::I64(1));
    b.add_group(g.finish());
    b.write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .group("grp")
            .unwrap()
            .remove_attr("missing")
            .unwrap();
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("not found"), "got: {err}");
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn add_variable_length_root_attribute_via_edit_session() {
    let path = temp_path("hdf5_pure_edit_vlen_group_attr.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .set_attr(
                "fields",
                AttrValue::VarLenAsciiCharArray(vec!["a".into(), "b".into()]),
            )
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert_eq!(
        attrs.get("fields"),
        Some(&AttrValue::VarLenAsciiCharArray(vec![
            "a".into(),
            "b".into()
        ]))
    );
    // The rest of the file is untouched.
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn add_variable_length_group_attribute_then_remove_then_reset_in_one_commit() {
    // A Set/Remove/Set sequence for the same name in one commit must leave
    // only the final value, whether or not the intermediate states are
    // variable-length — exercising `apply_compact_attr_ops`'s pending-VL-attr
    // bookkeeping (a plain region edit alone cannot represent an unresolved
    // variable-length attribute).
    let path = temp_path("hdf5_pure_edit_vlen_group_attr_sequence.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("grp");
    g.set_attr(
        "fields",
        AttrValue::VarLenAsciiCharArray(vec!["old1".into(), "old2".into()]),
    );
    b.add_group(g.finish());
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        // Replace the existing variable-length attribute with a fixed-size
        // one, then remove it, then set a fresh variable-length value.
        session
            .group("grp")
            .unwrap()
            .set_attr("fields", AttrValue::I64(1))
            .unwrap();
        session.group("grp").unwrap().remove_attr("fields").unwrap();
        session
            .group("grp")
            .unwrap()
            .set_attr(
                "fields",
                AttrValue::VarLenAsciiCharArray(vec!["new1".into(), "new2".into(), "new3".into()]),
            )
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let attrs = file.root().group("grp").unwrap().attrs().unwrap();
    assert_eq!(
        attrs.get("fields"),
        Some(&AttrValue::VarLenAsciiCharArray(vec![
            "new1".into(),
            "new2".into(),
            "new3".into()
        ]))
    );
}

/// A `Set` with a variable-length value must correctly drop a *fixed-size*
/// on-disk attribute of the same name, not just an existing pending
/// variable-length one: `apply_compact_attr_ops`'s `remove_attr_from_region`
/// call is otherwise only exercised by the plain `Remove` op, never by a
/// variable-length `Set` replacing a fixed-size value.
#[test]
fn set_variable_length_group_attribute_over_existing_fixed_attribute_in_one_commit() {
    let path = temp_path("hdf5_pure_edit_vlen_group_attr_over_fixed.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("grp");
    g.set_attr("fields", AttrValue::I64(42));
    b.add_group(g.finish());
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .group("grp")
            .unwrap()
            .set_attr(
                "fields",
                AttrValue::VarLenAsciiCharArray(vec!["new1".into(), "new2".into()]),
            )
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let attrs = file.root().group("grp").unwrap().attrs().unwrap();
    // Exactly one "fields" attribute survives, holding the new value — not a
    // leftover fixed-size copy alongside a new variable-length one.
    assert_eq!(attrs.len(), 1);
    assert_eq!(
        attrs.get("fields"),
        Some(&AttrValue::VarLenAsciiCharArray(vec![
            "new1".into(),
            "new2".into()
        ]))
    );
}

/// The compact-attribute budget check counts *pending* variable-length
/// attributes alongside attributes already resolved into the region
/// (`compact_attr_count(&out)? + pending_vl.len()`); exactly at the boundary
/// (6 existing fixed + 2 new variable-length = 8 = `MAX_COMPACT_ATTRS`) must
/// still succeed, with every value intact.
#[test]
fn add_variable_length_group_attributes_at_budget_boundary_in_one_commit() {
    let path = temp_path("hdf5_pure_edit_vlen_group_attr_at_budget.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("grp");
    for i in 0..6i64 {
        g.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    b.add_group(g.finish());
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        // Two elements each. A single element would read back distinctly now
        // that the reader keeps the written shape, but this case is about the
        // budget boundary, and two elements per attribute is what reaches it.
        session
            .group("grp")
            .unwrap()
            .set_attr(
                "b0",
                AttrValue::VarLenAsciiCharArray(vec!["x0".into(), "x1".into()]),
            )
            .unwrap();
        session
            .group("grp")
            .unwrap()
            .set_attr(
                "b1",
                AttrValue::VarLenAsciiCharArray(vec!["y0".into(), "y1".into()]),
            )
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let attrs = file.root().group("grp").unwrap().attrs().unwrap();
    assert_eq!(attrs.len(), 8);
    for i in 0..6i64 {
        assert_eq!(attrs.get(&format!("a{i}")), Some(&AttrValue::I64(i)));
    }
    assert_eq!(
        attrs.get("b0"),
        Some(&AttrValue::VarLenAsciiCharArray(vec![
            "x0".into(),
            "x1".into()
        ]))
    );
    assert_eq!(
        attrs.get("b1"),
        Some(&AttrValue::VarLenAsciiCharArray(vec![
            "y0".into(),
            "y1".into()
        ]))
    );
}

/// One variable-length attribute past the boundary above (6 existing fixed +
/// 3 new variable-length = 9) sends the set to a fractal heap. Since the 6
/// existing attributes alone are under the budget, this specifically exercises
/// the `+ pending_vl.len()` term of the count: a regression there would leave the
/// set compact, which is what the heap assertion catches.
#[test]
fn add_variable_length_group_attributes_over_budget_use_a_heap() {
    let path = temp_path("hdf5_pure_edit_vlen_group_attr_over_budget.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("grp");
    for i in 0..6i64 {
        g.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    b.add_group(g.finish());
    b.write(&path).unwrap();
    assert!(!has_fractal_heap(&std::fs::read(&path).unwrap()));

    {
        let session = File::open_rw(&path).unwrap();
        for i in 0..3 {
            session
                .group("grp")
                .unwrap()
                .set_attr(
                    &format!("b{i}"),
                    AttrValue::VarLenAsciiCharArray(vec![format!("x{i}")]),
                )
                .unwrap();
        }
        session.commit().unwrap();
    }

    assert!(
        has_fractal_heap(&std::fs::read(&path).unwrap()),
        "six fixed plus three variable-length attributes are past the compact budget",
    );
    let file = File::open(&path).unwrap();
    let attrs = file.group("grp").unwrap().attrs().unwrap();
    assert_eq!(attrs.len(), 9);
    for i in 0..6i64 {
        assert_eq!(attrs.get(&format!("a{i}")), Some(&AttrValue::I64(i)));
    }
    for i in 0..3 {
        assert_eq!(
            attrs.get(&format!("b{i}")),
            Some(&AttrValue::VarLenAsciiCharArray(vec![format!("x{i}")])),
            "a variable-length attribute rebuilt into the heap kept its strings",
        );
    }
}

/// A group already storing its attributes in a fractal heap takes an edit by
/// rebuilding that heap, variable-length attributes included (issue #102): the
/// variable-length `Set` path added for issue #105 has to reach the same rebuild
/// the fixed-size one does.
#[test]
fn dense_group_attribute_storage_takes_a_variable_length_edit() {
    let path = temp_path("hdf5_pure_edit_dense_group_attr.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("grp");
    for i in 0..12 {
        g.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    b.add_group(g.finish());
    b.write(&path).unwrap();
    assert!(has_fractal_heap(&std::fs::read(&path).unwrap()));

    {
        let session = File::open_rw(&path).unwrap();
        session
            .group("grp")
            .unwrap()
            .set_attr(
                "fields",
                AttrValue::VarLenAsciiCharArray(vec!["a".into(), "b".into()]),
            )
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let attrs = file.group("grp").unwrap().attrs().unwrap();
    assert_eq!(attrs.len(), 13);
    assert_eq!(
        attrs.get("fields"),
        Some(&AttrValue::VarLenAsciiCharArray(vec![
            "a".into(),
            "b".into()
        ])),
    );
    for i in 0..12 {
        assert_eq!(attrs.get(&format!("a{i}")), Some(&AttrValue::I64(i)));
    }
}

#[test]
fn deleting_group_with_attribute_edit_is_rejected_without_writing() {
    let path = temp_path("hdf5_pure_edit_delete_group_attr_overlap.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("grp");
    g.set_attr("tag", AttrValue::I64(1));
    b.add_group(g.finish());
    b.write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("grp").unwrap();
        session
            .group("grp")
            .unwrap()
            .set_attr("tag", AttrValue::I64(2))
            .unwrap();
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("overlaps"), "got: {err}");
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn copy_preserves_dataset_attributes() {
    // Exercises the "verbatim message bytes" claim: a copied dataset's
    // attributes (separate header messages) must survive byte-for-byte.
    let path = temp_path("hdf5_pure_edit_copy_attrs.h5");
    let mut b = FileBuilder::new();
    let ds = b.create_dataset("src");
    ds.with_i32_data(&[5, 6, 7]);
    ds.set_attr("label", AttrValue::String("alpha".into()));
    ds.set_attr("scale", AttrValue::F64(2.5));
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session.copy("src", "dup").unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let src_attrs = file.dataset("src").unwrap().attrs().unwrap();
    let dup = file.dataset("dup").unwrap();
    assert_eq!(dup.read_i32().unwrap(), vec![5, 6, 7]);
    // The copy's attributes equal the source's.
    assert_eq!(dup.attrs().unwrap(), src_attrs);
    assert_eq!(
        dup.attrs().unwrap().get("label"),
        Some(&AttrValue::String("alpha".into()))
    );
}

/// An unfiltered 2-D chunked dataset is copied: the values round-trip and the
/// copy is still chunked (the index is rebuilt at the new location).
#[test]
fn copy_unfiltered_chunked_dataset() {
    let path = temp_path("hdf5_pure_edit_copy_chunked.h5");
    let data: Vec<i32> = (0..24).collect();
    {
        let mut b = FileBuilder::new();
        b.create_dataset("src")
            .with_i32_data(&data)
            .with_shape(&[4, 6])
            .with_chunks(&[2, 3]);
        b.write(&path).unwrap();
    }
    {
        let session = File::open_rw(&path).unwrap();
        session.copy("src", "dup").unwrap();
        session.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    // Source untouched.
    assert_eq!(file.dataset("src").unwrap().read_i32().unwrap(), data);
    let dup = file.dataset("dup").unwrap();
    assert_eq!(dup.shape().unwrap(), vec![4, 6]);
    assert_eq!(dup.read_i32().unwrap(), data);
    assert!(
        dup.chunk_cache_stats().index_loaded(),
        "copied dataset must still be chunked"
    );
}

/// A filtered (shuffle + deflate) chunked dataset is copied verbatim: the chunk
/// bytes and filter pipeline are preserved (no recompression), the values round-
/// trip, the filter survives (the file stays far smaller than the raw bytes), and
/// the dataset's attributes are carried over.
#[test]
fn copy_filtered_chunked_dataset_preserves_pipeline_and_attrs() {
    let path = temp_path("hdf5_pure_edit_copy_filtered_chunked.h5");
    let data: Vec<i32> = (0..4096).map(|i| i % 4).collect(); // highly compressible
    {
        let mut b = FileBuilder::new();
        let ds = b.create_dataset("src");
        ds.with_i32_data(&data)
            .with_shape(&[4096])
            .with_chunks(&[512])
            .with_shuffle()
            .with_deflate(6);
        ds.set_attr("units", AttrValue::String("counts".into()));
        b.write(&path).unwrap();
    }
    {
        let session = File::open_rw(&path).unwrap();
        session.copy("src", "dup").unwrap();
        session.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    let dup = file.dataset("dup").unwrap();
    assert_eq!(dup.read_i32().unwrap(), data);
    assert!(
        dup.chunk_cache_stats().index_loaded(),
        "copied dataset must still be chunked"
    );
    // The filter survived: the whole file is far smaller than the raw element
    // bytes of a single copy, let alone two.
    assert!(
        std::fs::metadata(&path).unwrap().len() < (4096 * 4) as u64,
        "deflate filter must survive the copy"
    );
    // Attributes were preserved (the header is kept verbatim except its layout).
    assert_eq!(
        dup.attrs().unwrap().get("units"),
        Some(&AttrValue::String("counts".into()))
    );
}

/// An extensible (unlimited-dimension) chunked dataset copied within the file
/// stays readable; the copy uses an Extensible-Array index (selected from the
/// source's unlimited maxshape).
#[test]
fn copy_extensible_chunked_dataset() {
    let path = temp_path("hdf5_pure_edit_copy_extensible.h5");
    let data: Vec<f64> = (0..80).map(|i| i as f64 * 0.25).collect();
    write_starter(&path);
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("src", |b| {
                b.with_f64_data(&data)
                    .with_shape(&[80])
                    .with_chunks(&[16])
                    .with_maxshape(&[u64::MAX]);
            })
            .unwrap();
        session.commit().unwrap();
    }
    {
        let session = File::open_rw(&path).unwrap();
        session.copy("src", "dup").unwrap();
        session.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    let dup = file.dataset("dup").unwrap();
    assert_eq!(dup.read_f64().unwrap(), data);
    assert!(dup.chunk_cache_stats().index_loaded());
}

/// A single-chunk dataset is copied (the chunk address lives in the layout
/// message; the verbatim path re-emits a single-chunk layout).
#[test]
fn copy_single_chunk_dataset() {
    let path = temp_path("hdf5_pure_edit_copy_single_chunk.h5");
    let data: Vec<i32> = (0..16).collect();
    {
        let mut b = FileBuilder::new();
        b.create_dataset("src")
            .with_i32_data(&data)
            .with_shape(&[16])
            .with_chunks(&[16]); // one chunk covers the whole dataset
        b.write(&path).unwrap();
    }
    {
        let session = File::open_rw(&path).unwrap();
        session.copy("src", "dup").unwrap();
        session.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("dup").unwrap().read_i32().unwrap(), data);
}

#[test]
fn edit_preserves_multiple_root_datasets() {
    let path = temp_path("hdf5_pure_edit_multi_root.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d0").with_i32_data(&[0]);
    b.create_dataset("d1").with_i32_data(&[1]);
    b.create_dataset("d2").with_i32_data(&[2]);
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session.root().create_group("extra").unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let mut names = file.root().datasets().unwrap();
    names.sort();
    assert_eq!(names, vec!["d0", "d1", "d2"]);
    for (i, n) in ["d0", "d1", "d2"].iter().enumerate() {
        assert_eq!(file.dataset(n).unwrap().read_i32().unwrap(), vec![i as i32]);
    }
    assert_eq!(file.root().groups().unwrap(), vec!["extra".to_string()]);
}

#[test]
fn mixed_add_delete_copy_in_one_commit() {
    let path = temp_path("hdf5_pure_edit_mixed.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 1]);
    b.create_dataset("remove").with_i32_data(&[9]);
    b.create_dataset("source").with_f64_data(&[3.0, 3.0, 3.0]);
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("fresh", |b| {
                b.with_i32_data(&[42]);
            })
            .unwrap(); // add
        session.root().delete("remove").unwrap(); // delete
        session.copy("source", "source_copy").unwrap(); // copy
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let mut names = file.root().datasets().unwrap();
    names.sort();
    assert_eq!(names, vec!["fresh", "keep", "source", "source_copy"]);
    assert!(file.dataset("remove").is_err());
    assert_eq!(file.dataset("fresh").unwrap().read_i32().unwrap(), vec![42]);
    assert_eq!(
        file.dataset("source_copy").unwrap().read_f64().unwrap(),
        vec![3.0, 3.0, 3.0]
    );
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 1]
    );
}

#[test]
fn copy_from_file_dataset() {
    // Cross-file H5Ocopy: copy a dataset out of a separate open file.
    let src_path = temp_path("hdf5_pure_xcopy_src_ds.h5");
    let dst_path = temp_path("hdf5_pure_xcopy_dst_ds.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("payload").with_f64_data(&[1.5, 2.5, 3.5]);
        b.write(&src_path).unwrap();
    }
    write_starter(&dst_path);
    let src_bytes_before = std::fs::read(&src_path).unwrap();

    {
        let source = File::open(&src_path).unwrap();
        let session = File::open_rw(&dst_path).unwrap();
        session.copy_from(&source, "payload", "imported").unwrap();
        session.commit().unwrap();
    }

    // The copy landed in the destination, byte-equal to the source data.
    let file = File::open(&dst_path).unwrap();
    assert_eq!(
        file.dataset("imported").unwrap().read_f64().unwrap(),
        vec![1.5, 2.5, 3.5]
    );
    assert_eq!(
        file.dataset("imported").unwrap().dtype().unwrap(),
        DType::F64
    );
    // The destination's pre-existing dataset is untouched.
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    // The source file was not modified at all.
    assert_eq!(std::fs::read(&src_path).unwrap(), src_bytes_before);
}

#[test]
fn copy_from_file_group_subtree() {
    // A whole group subtree copied across files keeps its deep structure.
    let src_path = temp_path("hdf5_pure_xcopy_src_grp.h5");
    let dst_path = temp_path("hdf5_pure_xcopy_dst_grp.h5");
    write_starter(&src_path);
    {
        // Build the nested source subtree (FileBuilder::create_dataset does not
        // split paths into groups, so create the hierarchy explicitly).
        let s = File::open_rw(&src_path).unwrap();
        s.root().create_group("template").unwrap();
        s.root().create_group("template/inner").unwrap();
        s.root()
            .create_dataset("template/a", |b| {
                b.with_i32_data(&[1, 2]);
            })
            .unwrap();
        s.root()
            .create_dataset("template/inner/b", |b| {
                b.with_f64_data(&[9.0]);
            })
            .unwrap();
        s.commit().unwrap();
    }
    write_starter(&dst_path);

    {
        let source = File::open(&src_path).unwrap();
        let session = File::open_rw(&dst_path).unwrap();
        session.copy_from(&source, "template", "run1").unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&dst_path).unwrap();
    assert_eq!(
        file.dataset("run1/a").unwrap().read_i32().unwrap(),
        vec![1, 2]
    );
    assert_eq!(
        file.dataset("run1/inner/b").unwrap().read_f64().unwrap(),
        vec![9.0]
    );
    assert_eq!(
        file.group("run1").unwrap().groups().unwrap(),
        vec!["inner".to_string()]
    );
}

#[test]
fn copy_from_file_into_subgroup_created_same_session() {
    // The destination parent may be a group created earlier in this session.
    let src_path = temp_path("hdf5_pure_xcopy_src_into.h5");
    let dst_path = temp_path("hdf5_pure_xcopy_dst_into.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("payload").with_i32_data(&[7, 8, 9]);
        b.write(&src_path).unwrap();
    }
    write_starter(&dst_path);

    {
        let source = File::open(&src_path).unwrap();
        let session = File::open_rw(&dst_path).unwrap();
        session.root().create_group("dest").unwrap();
        session
            .copy_from(&source, "payload", "dest/payload_copy")
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&dst_path).unwrap();
    assert_eq!(
        file.dataset("dest/payload_copy")
            .unwrap()
            .read_i32()
            .unwrap(),
        vec![7, 8, 9]
    );
}

#[test]
fn copy_from_file_preserves_attributes() {
    // Fixed-size attributes survive a cross-file copy byte-for-byte.
    let src_path = temp_path("hdf5_pure_xcopy_src_attrs.h5");
    let dst_path = temp_path("hdf5_pure_xcopy_dst_attrs.h5");
    {
        let mut b = FileBuilder::new();
        let ds = b.create_dataset("src");
        ds.with_i32_data(&[5, 6, 7]);
        ds.set_attr("label", AttrValue::String("alpha".into()));
        ds.set_attr("scale", AttrValue::F64(2.5));
        b.write(&src_path).unwrap();
    }
    write_starter(&dst_path);

    let src_attrs = {
        let source = File::open(&src_path).unwrap();
        let attrs = source.dataset("src").unwrap().attrs().unwrap();
        let session = File::open_rw(&dst_path).unwrap();
        session.copy_from(&source, "src", "dup").unwrap();
        session.commit().unwrap();
        attrs
    };

    let file = File::open(&dst_path).unwrap();
    let dup = file.dataset("dup").unwrap();
    assert_eq!(dup.read_i32().unwrap(), vec![5, 6, 7]);
    assert_eq!(dup.attrs().unwrap(), src_attrs);
}

#[test]
fn copy_from_file_rejects_variable_length() {
    // A variable-length attribute stores global-heap references into the source
    // file; a verbatim cross-file copy cannot translate them, so it is refused.
    let src_path = temp_path("hdf5_pure_xcopy_src_vlen.h5");
    let dst_path = temp_path("hdf5_pure_xcopy_dst_vlen.h5");
    {
        let mut b = FileBuilder::new();
        let ds = b.create_dataset("src");
        ds.with_i32_data(&[1, 2, 3]);
        ds.set_attr(
            "tags",
            AttrValue::VarLenAsciiCharArray(vec!["one".into(), "two".into()]),
        );
        b.write(&src_path).unwrap();
    }
    write_starter(&dst_path);
    let dst_before = std::fs::read(&dst_path).unwrap();

    {
        let source = File::open(&src_path).unwrap();
        let session = File::open_rw(&dst_path).unwrap();
        let err = session.copy_from(&source, "src", "dup").unwrap_err();
        assert!(
            err.to_string().contains("variable-length or reference"),
            "got: {err}"
        );
        // Nothing was staged successfully, so a commit is a no-op.
        session.commit().unwrap();
    }

    // The destination is byte-unchanged; the same-file `copy` would have allowed
    // this (shared heap), but the cross-file path refuses it up front.
    assert_eq!(std::fs::read(&dst_path).unwrap(), dst_before);
}

#[test]
fn copy_from_file_rejects_reference_dataset() {
    // An object-reference dataset stores absolute source-file object addresses; a
    // verbatim cross-file copy cannot translate them. This exercises the
    // datatype-message refusal branch (the variable-length test above covers the
    // attribute branch).
    let src_path = temp_path("hdf5_pure_xcopy_src_ref.h5");
    let dst_path = temp_path("hdf5_pure_xcopy_dst_ref.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("target").with_i32_data(&[1, 2, 3]);
        b.create_dataset("refs").with_path_references(&["target"]);
        b.write(&src_path).unwrap();
    }
    write_starter(&dst_path);
    let dst_before = std::fs::read(&dst_path).unwrap();

    {
        let source = File::open(&src_path).unwrap();
        let session = File::open_rw(&dst_path).unwrap();
        let err = session.copy_from(&source, "refs", "dup").unwrap_err();
        assert!(
            err.to_string().contains("variable-length or reference"),
            "got: {err}"
        );
    }
    assert_eq!(std::fs::read(&dst_path).unwrap(), dst_before);
}

#[test]
fn copy_from_file_rejects_missing_source() {
    let src_path = temp_path("hdf5_pure_xcopy_src_missing.h5");
    let dst_path = temp_path("hdf5_pure_xcopy_dst_missing.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("present").with_i32_data(&[1]);
        b.write(&src_path).unwrap();
    }
    write_starter(&dst_path);

    let source = File::open(&src_path).unwrap();
    let session = File::open_rw(&dst_path).unwrap();
    let err = session.copy_from(&source, "ghost", "x").unwrap_err();
    assert!(err.to_string().contains("does not exist"), "got: {err}");
}

#[test]
fn copy_from_file_rejects_destination_collision() {
    // A destination name already present in the parent group is refused at commit,
    // leaving the file untouched.
    let src_path = temp_path("hdf5_pure_xcopy_src_collide.h5");
    let dst_path = temp_path("hdf5_pure_xcopy_dst_collide.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("payload").with_i32_data(&[1]);
        b.write(&src_path).unwrap();
    }
    write_starter(&dst_path); // contains "original"
    let dst_before = std::fs::read(&dst_path).unwrap();

    {
        let source = File::open(&src_path).unwrap();
        let session = File::open_rw(&dst_path).unwrap();
        session.copy_from(&source, "payload", "original").unwrap();
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("already exists"), "got: {err}");
    }
    assert_eq!(std::fs::read(&dst_path).unwrap(), dst_before);
}

#[test]
fn copy_from_file_rejects_streaming_source() {
    // The source must be buffered so its bytes are addressable; a streaming reader
    // is refused with a clear message.
    let src_path = temp_path("hdf5_pure_xcopy_src_stream.h5");
    let dst_path = temp_path("hdf5_pure_xcopy_dst_stream.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("payload").with_i32_data(&[1, 2, 3]);
        b.write(&src_path).unwrap();
    }
    write_starter(&dst_path);

    let source = File::open_streaming(&src_path).unwrap();
    let session = File::open_rw(&dst_path).unwrap();
    let err = session.copy_from(&source, "payload", "dup").unwrap_err();
    assert!(err.to_string().contains("buffered source"), "got: {err}");
}

#[test]
fn copy_same_file_still_allows_variable_length_attribute() {
    // Regression guard: the foreign-address refusal is cross-file only. An in-file
    // `copy` of a dataset carrying a variable-length attribute still works (the
    // copy shares the source file's global heap).
    let path = temp_path("hdf5_pure_xcopy_infile_vlen.h5");
    {
        let mut b = FileBuilder::new();
        let ds = b.create_dataset("src");
        ds.with_i32_data(&[1, 2, 3]);
        ds.set_attr(
            "tags",
            AttrValue::VarLenAsciiCharArray(vec!["one".into(), "two".into()]),
        );
        b.write(&path).unwrap();
    }

    {
        let session = File::open_rw(&path).unwrap();
        session.copy("src", "dup").unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let src_attrs = file.dataset("src").unwrap().attrs().unwrap();
    assert_eq!(file.dataset("dup").unwrap().attrs().unwrap(), src_attrs);
    assert_eq!(
        file.dataset("dup").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
}

#[test]
fn superblock_eof_matches_file_size_after_edit() {
    let path = temp_path("hdf5_pure_edit_eof.h5");
    write_starter(&path);
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("more", |b| {
                b.with_f64_data(&[1.0, 2.0, 3.0]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let on_disk = std::fs::metadata(&path).unwrap().len();
    // The edit updates the superblock's logical end-of-file to the new size.
    assert_eq!(file.file_size(), on_disk);
    assert_eq!(file.superblock().eof_address, on_disk);
}

/// A chunked (but unfiltered) dataset can be added in place and read back, and
/// the original dataset is left intact.
#[test]
fn add_chunked_dataset() {
    let path = temp_path("hdf5_pure_edit_add_chunked.h5");
    write_starter(&path);

    let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.5).collect();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("chunky", |b| {
                b.with_f64_data(&data).with_chunks(&[25]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let chunky = file.dataset("chunky").unwrap();
    assert_eq!(chunky.shape().unwrap(), vec![100]);
    assert_eq!(chunky.read_f64().unwrap(), data);
    // Original dataset untouched.
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    // The superblock's end-of-file matches the physical size after appending the
    // chunk data, index, and header.
    assert_eq!(file.file_size(), std::fs::metadata(&path).unwrap().len());
}

/// Deflate, shuffle+deflate, and fletcher32 filtered datasets each round-trip
/// through the in-place editor and the reader.
#[test]
fn add_filtered_datasets() {
    let path = temp_path("hdf5_pure_edit_add_filtered.h5");
    write_starter(&path);

    let data: Vec<f64> = (0..200).map(|i| i as f64).collect();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("deflated", |b| {
                b.with_f64_data(&data).with_chunks(&[50]).with_deflate(6);
            })
            .unwrap();
        session
            .root()
            .create_dataset("shuffled", |b| {
                b.with_f64_data(&data)
                    .with_chunks(&[50])
                    .with_shuffle()
                    .with_deflate(4);
            })
            .unwrap();
        session
            .root()
            .create_dataset("checked", |b| {
                b.with_f64_data(&data).with_chunks(&[64]).with_fletcher32();
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    for name in ["deflated", "shuffled", "checked"] {
        assert_eq!(
            file.dataset(name).unwrap().read_f64().unwrap(),
            data,
            "dataset {name} did not round-trip"
        );
    }
    assert_eq!(file.file_size(), std::fs::metadata(&path).unwrap().len());
}

/// A lossless integer scale-offset dataset round-trips.
#[test]
fn add_scale_offset_dataset() {
    let path = temp_path("hdf5_pure_edit_add_scaleoffset.h5");
    write_starter(&path);

    let data: Vec<i32> = (0..120).map(|i| 1000 + (i % 7)).collect();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("counts", |b| {
                b.with_i32_data(&data)
                    .with_chunks(&[40])
                    .with_scale_offset(ScaleOffset::Integer(0));
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("counts").unwrap().read_i32().unwrap(), data);
}

/// A 2-D chunked dataset whose chunks don't evenly divide the shape round-trips
/// (exercises edge chunks and the fixed-array index used for >1 chunk).
#[test]
fn add_2d_chunked_dataset() {
    let path = temp_path("hdf5_pure_edit_add_2d_chunked.h5");
    write_starter(&path);

    let data: Vec<i32> = (0..(7 * 5)).collect();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("grid", |b| {
                b.with_i32_data(&data)
                    .with_shape(&[7, 5])
                    .with_chunks(&[3, 2]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let grid = file.dataset("grid").unwrap();
    assert_eq!(grid.shape().unwrap(), vec![7, 5]);
    assert_eq!(grid.read_i32().unwrap(), data);
}

/// An extensible (unlimited-dimension) dataset can be added in place; its data
/// reads back and the file remains valid. The unlimited dimension selects the
/// Extensible-Array chunk index.
#[test]
fn add_extensible_dataset() {
    let path = temp_path("hdf5_pure_edit_add_extensible.h5");
    write_starter(&path);

    let data: Vec<i32> = (0..64).collect();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("stream", |b| {
                b.with_i32_data(&data)
                    .with_shape(&[64])
                    .with_maxshape(&[u64::MAX])
                    .with_chunks(&[16]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let stream = file.dataset("stream").unwrap();
    assert_eq!(stream.shape().unwrap(), vec![64]);
    assert_eq!(stream.read_i32().unwrap(), data);
    assert_eq!(file.file_size(), std::fs::metadata(&path).unwrap().len());
}

/// One commit can mix a contiguous dataset and a chunked/compressed dataset
/// into a nested group, alongside the original.
#[test]
fn add_mixed_contiguous_and_chunked_in_group() {
    let path = temp_path("hdf5_pure_edit_add_mixed.h5");
    write_starter(&path);

    let wave: Vec<f64> = (0..512).map(|i| (i as f64 * 0.1).cos()).collect();
    {
        let session = File::open_rw(&path).unwrap();
        session.root().create_group("run").unwrap();
        session
            .root()
            .create_dataset("run/scalarish", |b| {
                b.with_i32_data(&[1, 2, 3]);
            })
            .unwrap();
        session
            .root()
            .create_dataset("run/wave", |b| {
                b.with_f64_data(&wave)
                    .with_chunks(&[128])
                    .with_shuffle()
                    .with_deflate(6);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("run/scalarish").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(file.dataset("run/wave").unwrap().read_f64().unwrap(), wave);
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(file.file_size(), std::fs::metadata(&path).unwrap().len());
}

/// A chunked dataset whose datatype is `f64` still reports the right dtype after
/// an in-place add, confirming the header is a faithful chunked dataset header.
#[test]
fn added_chunked_dataset_reports_dtype() {
    let path = temp_path("hdf5_pure_edit_chunked_dtype.h5");
    write_starter(&path);
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("c", |b| {
                b.with_f64_data(&(0..50).map(f64::from).collect::<Vec<_>>())
                    .with_chunks(&[10])
                    .with_deflate(3);
            })
            .unwrap();
        session.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("c").unwrap().dtype().unwrap(), DType::F64);
}

/// A ZFP fixed-rate compressed dataset can be added in place and reads back
/// within ZFP's lossy tolerance (gated on the `zfp` feature).
#[test]
#[cfg(feature = "zfp")]
fn add_zfp_dataset() {
    let path = temp_path("hdf5_pure_edit_add_zfp.h5");
    write_starter(&path);

    let data: Vec<f64> = (0..256).map(|i| (i as f64 * 0.05).sin()).collect();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("zfp", |b| {
                b.with_f64_data(&data).with_chunks(&[64]).with_zfp(32.0);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let back = file.dataset("zfp").unwrap().read_f64().unwrap();
    assert_eq!(back.len(), data.len());
    let max_err = data
        .iter()
        .zip(back.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0f64, f64::max);
    assert!(max_err < 1e-6, "ZFP max_err {max_err} > 1e-6");
}

/// Malformed chunked-dataset requests are refused before any byte is written,
/// rather than panicking or producing a silently-corrupt dataset: chunk dims
/// whose rank disagrees with the shape, a zero chunk dimension, a maxshape whose
/// rank disagrees with the shape or falls below it, a chunked scalar, and a
/// zero-element shape left to auto-chunking (which would resolve to a zero
/// chunk dimension and divide by zero in the splitter — an *explicitly* chunked
/// zero-element shape is legal, see
/// `add_empty_extensible_chunked_dataset_and_grow_it`).
///
/// Each case names the substring its own refusal must carry: without that, a
/// case refused for some unrelated reason still passes, and the guard under test
/// is never reached.
#[test]
fn malformed_chunked_requests_are_rejected_without_writing() {
    // A no-capture configurator per malformed case; `fn` pointers keep the case
    // table a simple type.
    type Configure = fn(&mut hdf5_pure::DatasetBuilder);
    let bad: &[(&str, Configure, &str)] = &[
        (
            "auto-chunked empty shape",
            |b| {
                b.with_f64_data(&[])
                    .with_shape(&[0])
                    .with_maxshape(&[u64::MAX]);
            },
            "explicit chunk dimensions",
        ),
        (
            "chunk rank mismatch",
            |b| {
                b.with_i32_data(&[1, 2, 3, 4, 5, 6])
                    .with_shape(&[2, 3])
                    .with_chunks(&[2]);
            },
            "chunk dimensions must have the same rank",
        ),
        (
            "zero chunk dim",
            |b| {
                b.with_i32_data(&[1, 2, 3, 4])
                    .with_shape(&[4])
                    .with_chunks(&[0]);
            },
            "chunk dimensions must all be non-zero",
        ),
        (
            "maxshape rank mismatch",
            |b| {
                b.with_i32_data(&[1, 2, 3, 4])
                    .with_shape(&[4])
                    .with_maxshape(&[u64::MAX, u64::MAX])
                    .with_chunks(&[2]);
            },
            "maxshape must have the same rank",
        ),
        (
            "scalar with chunks",
            |b| {
                b.with_f64_data(&[1.0]).with_shape(&[]).with_chunks(&[1]);
            },
            "a scalar dataset cannot be chunked",
        ),
        (
            "maxshape below shape",
            |b| {
                b.with_i32_data(&[1, 2, 3, 4])
                    .with_shape(&[4])
                    .with_maxshape(&[2]);
            },
            "maxshape must be at least the current shape",
        ),
    ];

    for (label, configure, expected) in bad {
        let path = temp_path(&format!(
            "hdf5_pure_edit_reject_{}.h5",
            label.replace(' ', "_")
        ));
        write_starter(&path);
        let before = std::fs::read(&path).unwrap();
        {
            let session = File::open_rw(&path).unwrap();
            let err = session.root().create_dataset("bad", configure).unwrap_err();
            let text = err.to_string();
            assert!(
                text.contains("in-place edit"),
                "[{label}] expected an EditUnsupported refusal, got: {err}"
            );
            assert!(
                text.contains(expected),
                "[{label}] refusal must name {expected:?}, got: {err}"
            );
            session.commit().unwrap();
        }
        // The guard runs as the dataset is staged, so the file is untouched.
        assert_eq!(
            std::fs::read(&path).unwrap(),
            before,
            "[{label}] file modified"
        );
    }
}

// ---- write_dataset: in-place value overwrite (issue #79) ----

#[test]
fn write_dataset_same_size_overwrites_in_place() {
    let path = temp_path("hdf5_pure_write_same_size.h5");
    write_starter(&path); // "original" = [1.0, 2.0, 3.0, 4.0] (contiguous f64)
    let size_before = std::fs::metadata(&path).unwrap().len();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("original")
            .unwrap()
            .write_staged(|b| {
                b.with_f64_data(&[9.0, 8.0, 7.0, 6.0]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    // Same-length overwrite reuses the existing block: the file does not grow.
    let size_after = std::fs::metadata(&path).unwrap().len();
    assert_eq!(
        size_after, size_before,
        "same-size write should not grow file"
    );

    let file = File::open(&path).unwrap();
    let ds = file.dataset("original").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::F64);
    assert_eq!(ds.read_f64().unwrap(), vec![9.0, 8.0, 7.0, 6.0]);
}

#[test]
fn write_dataset_resize_keeping_shape_is_a_reshape_and_refused() {
    // For a fixed-size datatype the byte length is shape * element size, so a
    // different-length replacement necessarily changes the shape — which is a
    // reshape, not a value overwrite, and is refused. (The genuine relocation
    // path — overwriting a never-written, undefined-address dataset — is exercised
    // in the crosscheck against the C library, which can create one.)
    let path = temp_path("hdf5_pure_write_resize_refused.h5");
    write_starter(&path); // "original" = 4 f64
    let before = std::fs::read(&path).unwrap();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("original")
            .unwrap()
            .write_staged(|b| {
                b.with_f64_data(&[42.0]);
            })
            .unwrap();
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string().contains("shape does not match"),
            "expected reshape refusal, got: {err}"
        );
    }
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "file modified on refusal"
    );
}

#[test]
fn write_dataset_in_a_nested_group() {
    let path = temp_path("hdf5_pure_write_nested.h5");
    {
        let mut b = FileBuilder::new();
        let mut g = b.create_group("grp");
        g.create_dataset("inner").with_i32_data(&[1, 2, 3]);
        b.add_group(g.finish());
        b.write(&path).unwrap();
    }
    {
        let session = File::open_rw(&path).unwrap();
        // Same size (in place) for the nested dataset.
        session
            .dataset("grp/inner")
            .unwrap()
            .write_staged(|b| {
                b.with_i32_data(&[10, 20, 30]);
            })
            .unwrap();
        session.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("grp/inner").unwrap().read_i32().unwrap(),
        vec![10, 20, 30]
    );
}

#[test]
fn write_dataset_rejects_datatype_mismatch() {
    let path = temp_path("hdf5_pure_write_type_mismatch.h5");
    write_starter(&path); // f64
    let before = std::fs::read(&path).unwrap();
    {
        let session = File::open_rw(&path).unwrap();
        // i32 data for an f64 dataset: a retype, refused.
        session
            .dataset("original")
            .unwrap()
            .write_staged(|b| {
                b.with_i32_data(&[1, 2, 3, 4]);
            })
            .unwrap();
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string().contains("datatype does not match"),
            "expected datatype-mismatch refusal, got: {err}"
        );
    }
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "file modified on refusal"
    );
}

#[test]
fn write_dataset_rejects_shape_mismatch() {
    let path = temp_path("hdf5_pure_write_shape_mismatch.h5");
    {
        let mut b = FileBuilder::new();
        // A 2-D dataset, so a 1-D replacement of the same element count is a
        // reshape (different dataspace bytes), which is refused.
        b.create_dataset("m")
            .with_i32_data(&[1, 2, 3, 4, 5, 6])
            .with_shape(&[2, 3]);
        b.write(&path).unwrap();
    }
    let before = std::fs::read(&path).unwrap();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("m")
            .unwrap()
            .write_staged(|b| {
                b.with_i32_data(&[1, 2, 3, 4, 5, 6]).with_shape(&[6]);
            })
            .unwrap();
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string().contains("shape does not match"),
            "expected shape-mismatch refusal, got: {err}"
        );
    }
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "file modified on refusal"
    );
}

#[test]
fn write_dataset_rejects_missing_target() {
    let path = temp_path("hdf5_pure_write_missing.h5");
    write_starter(&path);
    {
        let session = File::open_rw(&path).unwrap();
        // The missing target is now reported when the handle is resolved, before
        // an overwrite can be staged, rather than at commit.
        let err = session.dataset("nope").unwrap_err();
        assert!(
            matches!(err, Error::Format(FormatError::PathNotFound(ref p)) if p == "nope"),
            "expected PathNotFound naming the missing dataset, got: {err:?}"
        );
        assert!(!session.has_staged_edits());
    }
}

/// An unfiltered chunked dataset is overwritten chunk-by-chunk straight in its
/// existing slots: the file does not grow (no header rewrite, no index change)
/// and the new values read back.
#[test]
fn write_dataset_overwrites_unfiltered_chunked_in_place() {
    let path = temp_path("hdf5_pure_write_chunked.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("c")
            .with_i32_data(&[1, 2, 3, 4, 5, 6, 7, 8])
            .with_shape(&[8])
            .with_chunks(&[4]);
        b.write(&path).unwrap();
    }
    let size_before = std::fs::metadata(&path).unwrap().len();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("c")
            .unwrap()
            .write_staged(|b| {
                b.with_i32_data(&[8, 7, 6, 5, 4, 3, 2, 1]).with_shape(&[8]);
            })
            .unwrap();
        session.commit().unwrap();
    }
    // An unfiltered chunked overwrite is a true in-place write: the chunk slots
    // are reused, so the file did not grow.
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        size_before,
        "unfiltered chunked overwrite should not grow the file"
    );
    let file = File::open(&path).unwrap();
    let c = file.dataset("c").unwrap();
    assert_eq!(c.shape().unwrap(), vec![8]);
    assert_eq!(c.read_i32().unwrap(), vec![8, 7, 6, 5, 4, 3, 2, 1]);
    assert!(
        c.chunk_cache_stats().index_loaded(),
        "dataset must still be chunked"
    );
}

/// A 2-D chunked dataset whose chunks do not evenly divide the shape (edge
/// chunks, Fixed-Array index) is overwritten in place.
#[test]
fn write_dataset_overwrites_2d_edge_chunked_in_place() {
    let path = temp_path("hdf5_pure_write_2d_edge_chunked.h5");
    let orig: Vec<i32> = (0..35).collect();
    {
        let mut b = FileBuilder::new();
        b.create_dataset("g")
            .with_i32_data(&orig)
            .with_shape(&[7, 5])
            .with_chunks(&[3, 2]);
        b.write(&path).unwrap();
    }
    let size_before = std::fs::metadata(&path).unwrap().len();
    let updated: Vec<i32> = orig.iter().rev().copied().collect();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("g")
            .unwrap()
            .write_staged(|b| {
                b.with_i32_data(&updated).with_shape(&[7, 5]);
            })
            .unwrap();
        session.commit().unwrap();
    }
    assert_eq!(std::fs::metadata(&path).unwrap().len(), size_before);
    let file = File::open(&path).unwrap();
    let g = file.dataset("g").unwrap();
    assert_eq!(g.shape().unwrap(), vec![7, 5]);
    assert_eq!(g.read_i32().unwrap(), updated);
}

/// An extensible (unlimited-dimension, Extensible-Array index) chunked dataset is
/// overwritten in place.
#[test]
fn write_dataset_overwrites_extensible_chunked_in_place() {
    let path = temp_path("hdf5_pure_write_extensible_chunked.h5");
    let orig: Vec<f64> = (0..60).map(|i| i as f64).collect();
    write_starter(&path);
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("ext", |b| {
                b.with_f64_data(&orig)
                    .with_shape(&[60])
                    .with_chunks(&[16])
                    .with_maxshape(&[u64::MAX]);
            })
            .unwrap();
        session.commit().unwrap();
    }
    let updated: Vec<f64> = orig.iter().map(|v| v * 2.0).collect();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("ext")
            .unwrap()
            .write_staged(|b| {
                b.with_f64_data(&updated).with_shape(&[60]);
            })
            .unwrap();
        session.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("ext").unwrap().read_f64().unwrap(), updated);
}

/// A size-preserving filter (Fletcher32 always appends a 4-byte checksum, so the
/// stored size is independent of the values) lets a filtered chunked dataset be
/// overwritten in place even when the values change.
#[test]
fn write_dataset_overwrites_fletcher32_chunked_in_place() {
    let path = temp_path("hdf5_pure_write_fletcher_chunked.h5");
    let orig: Vec<f64> = (0..128).map(|i| i as f64).collect();
    write_starter(&path);
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("ck", |b| {
                b.with_f64_data(&orig)
                    .with_shape(&[128])
                    .with_chunks(&[64])
                    .with_fletcher32();
            })
            .unwrap();
        session.commit().unwrap();
    }
    let size_before = std::fs::metadata(&path).unwrap().len();
    let updated: Vec<f64> = orig.iter().map(|v| v + 1000.0).collect();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("ck")
            .unwrap()
            .write_staged(|b| {
                b.with_f64_data(&updated).with_shape(&[128]);
            })
            .unwrap();
        session.commit().unwrap();
    }
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        size_before,
        "a Fletcher32 overwrite keeps each chunk's stored size, so it stays in place"
    );
    let file = File::open(&path).unwrap();
    let ck = file.dataset("ck").unwrap();
    assert_eq!(ck.read_f64().unwrap(), updated);
    assert!(ck.chunk_cache_stats().index_loaded());
}

/// Rewriting a deflate dataset with the *same* values reproduces identical
/// compressed bytes, so the overwrite fits the existing slots and stays in place.
#[test]
fn write_dataset_overwrites_deflate_chunked_equal_size_in_place() {
    let path = temp_path("hdf5_pure_write_deflate_equal.h5");
    let data: Vec<f64> = (0..200).map(|i| i as f64).collect();
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_f64_data(&data)
            .with_shape(&[200])
            .with_chunks(&[50])
            .with_deflate(6);
        b.write(&path).unwrap();
    }
    let size_before = std::fs::metadata(&path).unwrap().len();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("d")
            .unwrap()
            .write_staged(|b| {
                b.with_f64_data(&data).with_shape(&[200]);
            })
            .unwrap();
        session.commit().unwrap();
    }
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        size_before,
        "re-encoding identical values is byte-identical, so the overwrite stays in place"
    );
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_f64().unwrap(), data);
}

/// A deflate dataset overwritten with values of different compressibility
/// re-encodes to a different size, so the dataset is rebuilt and relocated; the
/// new values still read back and the dataset stays chunked + compressed.
#[test]
fn write_dataset_overwrites_deflate_chunked_relocates_on_size_change() {
    let path = temp_path("hdf5_pure_write_deflate_relocate.h5");
    // Highly compressible original, then incompressible-ish replacement.
    let orig: Vec<i32> = vec![0; 4096];
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&orig)
            .with_shape(&[4096])
            .with_chunks(&[512])
            .with_deflate(6);
        b.write(&path).unwrap();
    }
    let updated: Vec<i32> = (0..4096i32)
        .map(|i| i.wrapping_mul(2_654_435_761u32 as i32) ^ (i << 3))
        .collect();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("d")
            .unwrap()
            .write_staged(|b| {
                b.with_i32_data(&updated).with_shape(&[4096]);
            })
            .unwrap();
        session.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    let d = file.dataset("d").unwrap();
    assert_eq!(d.read_i32().unwrap(), updated);
    assert!(
        d.chunk_cache_stats().index_loaded(),
        "relocated dataset must still be chunked"
    );
}

/// A relocating chunked overwrite returns the old chunk storage to the session's
/// free list (the same path the delete reclaim uses), and a subsequent addition
/// in the same session draws from it. This exercises the relocate -> free ->
/// reuse interplay: the file must stay valid and both datasets read back exactly
/// (a double-free or stale span would corrupt one of them).
#[test]
fn write_dataset_chunked_relocate_then_reuse_stays_valid() {
    let path = temp_path("hdf5_pure_write_chunked_reclaim.h5");
    {
        let mut b = FileBuilder::new();
        // Highly compressible start => tiny chunk slots.
        b.create_dataset("d")
            .with_i32_data(&vec![0i32; 4096])
            .with_shape(&[4096])
            .with_chunks(&[512])
            .with_deflate(6);
        b.write(&path).unwrap();
    }
    // Incompressible new values grow the chunks past their tiny slots, forcing a
    // relocate that frees the old chunk storage into the session free list.
    let updated: Vec<i32> = (0..4096i32)
        .map(|i| i.wrapping_mul(2_654_435_761u32 as i32) ^ (i << 3))
        .collect();
    let filler: Vec<f64> = (0..64).map(|i| i as f64).collect();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("d")
            .unwrap()
            .write_staged(|b| {
                b.with_i32_data(&updated).with_shape(&[4096]);
            })
            .unwrap();
        session.commit().unwrap();

        // A later addition in the same session draws from the freed regions.
        session
            .root()
            .create_dataset("filler", |b| {
                b.with_f64_data(&filler);
            })
            .unwrap();
        session.commit().unwrap();
    } // drop the editor (release its file lock) before reading back

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), updated);
    assert_eq!(file.dataset("filler").unwrap().read_f64().unwrap(), filler);
}

/// A filtered (deflate, Fixed-Array index) overwrite whose re-encoded chunks are
/// *smaller* than their slots is applied in place: each shrunk chunk is written
/// into its existing slot and the chunk index is rebuilt in place to record the
/// new sizes, so the file does not grow and the new values read back (which would
/// be impossible if the index still recorded the old, larger sizes).
#[test]
fn write_dataset_overwrites_filtered_chunked_fits_with_slack_in_place() {
    let path = temp_path("hdf5_pure_write_fits_slack_fa.h5");
    // Incompressible start => large chunk slots (Fixed Array: 4 finite chunks).
    let orig: Vec<i32> = (0..2048i32)
        .map(|i| i.wrapping_mul(2_654_435_761u32 as i32) ^ (i << 3))
        .collect();
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&orig)
            .with_shape(&[2048])
            .with_chunks(&[512])
            .with_deflate(6);
        b.write(&path).unwrap();
    }
    let size_before = std::fs::metadata(&path).unwrap().len();
    // Highly compressible replacement => much smaller chunks that fit with slack.
    let updated: Vec<i32> = vec![7; 2048];
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
        "a fits-with-slack overwrite reuses the chunk slots and rebuilds the index \
         in place, so the file must not grow"
    );
    let file = File::open(&path).unwrap();
    let d = file.dataset("d").unwrap();
    assert_eq!(d.read_i32().unwrap(), updated);
    assert!(d.chunk_cache_stats().index_loaded(), "still chunked");
}

/// The fits-with-slack in-place path also covers an extensible (unlimited,
/// Extensible-Array index) dataset: the shrunk chunks reuse their slots and the
/// EA index is rebuilt in place, so the file does not grow and the values read
/// back.
#[test]
fn write_dataset_overwrites_filtered_extensible_fits_with_slack() {
    let path = temp_path("hdf5_pure_write_fits_slack_ea.h5");
    let orig: Vec<i32> = (0..2048i32)
        .map(|i| i.wrapping_mul(2_654_435_761u32 as i32) ^ (i << 3))
        .collect();
    write_starter(&path);
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("d", |b| {
                b.with_i32_data(&orig)
                    .with_shape(&[2048])
                    .with_chunks(&[512])
                    .with_maxshape(&[u64::MAX]) // unlimited => Extensible Array index
                    .with_deflate(6);
            })
            .unwrap();
        session.commit().unwrap();
    }
    let size_before = std::fs::metadata(&path).unwrap().len();
    let updated: Vec<i32> = vec![3; 2048];
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
        "an extensible-array fits-with-slack overwrite also rebuilds the index in \
         place, so the file must not grow"
    );
    let file = File::open(&path).unwrap();
    let d = file.dataset("d").unwrap();
    assert_eq!(d.read_i32().unwrap(), updated);
    assert!(d.chunk_cache_stats().index_loaded(), "still chunked");
}

#[test]
fn write_dataset_rejects_filtered_request() {
    // A builder that itself requests chunking/filtering is refused as "not a
    // value overwrite" before the on-disk dataset is even consulted — which is
    // why the refusal arrives from `write_staged` and not from the commit.
    let path = temp_path("hdf5_pure_write_filtered_request.h5");
    write_starter(&path);
    {
        let session = File::open_rw(&path).unwrap();
        let err = session
            .dataset("original")
            .unwrap()
            .write_staged(|b| {
                b.with_f64_data(&[1.0, 2.0, 3.0, 4.0])
                    .with_shape(&[4])
                    .with_chunks(&[2]);
            })
            .unwrap_err();
        assert!(
            err.to_string().contains("overwrites values only"),
            "expected value-only refusal, got: {err}"
        );
        assert!(
            !session.has_staged_edits(),
            "a refused overwrite must not be left staged"
        );
    }
}

#[test]
fn write_dataset_rejects_staged_attributes() {
    // Attributes set on the write_dataset builder cannot be applied by a value
    // overwrite, so they must be refused rather than silently dropped.
    let path = temp_path("hdf5_pure_write_attr_refused.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();
    {
        let session = File::open_rw(&path).unwrap();
        let err = session
            .dataset("original")
            .unwrap()
            .write_staged(|b| {
                b.with_f64_data(&[5.0, 6.0, 7.0, 8.0]) // same size, valid overwrite
                    .set_attr("units", AttrValue::String("m/s".into()));
            })
            .unwrap_err();
        assert!(
            err.to_string().contains("cannot set attributes"),
            "expected attribute refusal, got: {err}"
        );
        assert!(
            !session.has_staged_edits(),
            "a refused overwrite must not be left staged"
        );
    }
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "file modified on refusal"
    );
}

#[test]
fn write_dataset_alongside_other_edits() {
    // A value overwrite coexists with an addition and a delete in one commit.
    let path = temp_path("hdf5_pure_write_mixed.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("keep").with_f64_data(&[1.0, 2.0]);
        b.create_dataset("doomed").with_i32_data(&[9]);
        b.write(&path).unwrap();
    }
    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("keep")
            .unwrap()
            .write_staged(|b| {
                b.with_f64_data(&[5.0, 6.0]);
            })
            .unwrap(); // same size
        session
            .root()
            .create_dataset("added", |b| {
                b.with_i32_data(&[3, 4]);
            })
            .unwrap();
        session.root().delete("doomed").unwrap();
        session.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("keep").unwrap().read_f64().unwrap(),
        vec![5.0, 6.0]
    );
    assert_eq!(
        file.dataset("added").unwrap().read_i32().unwrap(),
        vec![3, 4]
    );
    assert!(file.dataset("doomed").is_err());
}

#[test]
fn write_dataset_with_no_other_edits_takes_inplace_fast_path() {
    // A lone same-size overwrite must not rewrite headers or flip the root: the
    // only on-disk bytes that change are the data block itself.
    let path = temp_path("hdf5_pure_write_fastpath.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("original")
            .unwrap()
            .write_staged(|b| {
                b.with_f64_data(&[1.0, 2.0, 3.0, 4.0]);
            })
            .unwrap(); // identical bytes
        session.commit().unwrap();
    }
    // Identical data written back in place leaves the file byte-for-byte the same.
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "in-place rewrite of identical data changed the file"
    );
}

/// A zero-element (empty) contiguous dataset — the on-disk equivalent of the
/// whole-file writer's `HADDR_UNDEF` data address for "no storage allocated"
/// (issue #105) — can be added in place, alongside an ordinary dataset in the
/// same commit.
#[test]
fn add_empty_dataset_via_edit_session() {
    let path = temp_path("hdf5_pure_edit_add_empty.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("empty", |b| {
                b.with_f64_data(&[]).with_shape(&[0, 3]);
            })
            .unwrap();
        session
            .root()
            .create_dataset("added", |b| {
                b.with_i32_data(&[7, 8]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let empty = file.dataset("empty").unwrap();
    assert_eq!(empty.shape().unwrap(), vec![0, 3]);
    assert_eq!(empty.dtype().unwrap(), DType::F64);
    assert_eq!(empty.read_f64().unwrap(), Vec::<f64>::new());
    assert_eq!(
        file.dataset("added").unwrap().read_i32().unwrap(),
        vec![7, 8]
    );
    // The original, pre-existing dataset is untouched.
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

/// An empty dataset whose supplied data does not match its (zero-element)
/// shape is rejected rather than silently written as unreachable, orphaned
/// storage: `flatten_dataset` must validate `raw` against the shape even when
/// the shape contains a `0` dimension, not just for non-empty shapes.
#[test]
fn add_empty_dataset_with_mismatched_data_is_rejected_without_writing() {
    let path = temp_path("hdf5_pure_edit_add_empty_mismatched.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        let err = session
            .root()
            .create_dataset("bogus", |b| {
                b.with_f64_data(&[9.0, 9.0, 9.0]).with_shape(&[0, 3]);
            })
            .unwrap_err();
        assert!(err.to_string().contains("shape"), "got: {err}");
        session.commit().unwrap();
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
}

/// A provenance-tagged dataset (issue #105) can be added in place; the
/// SHA-256/creator/timestamp/source attributes are computed and stored
/// exactly as the whole-file writer does, and `verify_provenance` confirms
/// the hash while the attribute values themselves are checked directly.
#[cfg(feature = "provenance")]
#[test]
fn add_provenance_dataset_via_edit_session() {
    use hdf5_pure::VerifyResult;

    let path = temp_path("hdf5_pure_edit_add_provenance.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("sensor", |b| {
                b.with_f64_data(&[1.0, 2.0, 3.0]).with_provenance(
                    "test-suite",
                    "2026-02-19T12:00:00Z",
                    Some("bench"),
                );
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("sensor").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(ds.verify_provenance().unwrap(), VerifyResult::Ok);
    let attrs = ds.attrs().unwrap();
    assert_eq!(
        attrs.get("_provenance_creator"),
        Some(&AttrValue::String("test-suite".into()))
    );
    assert_eq!(
        attrs.get("_provenance_timestamp"),
        Some(&AttrValue::String("2026-02-19T12:00:00Z".into()))
    );
    assert_eq!(
        attrs.get("_provenance_source"),
        Some(&AttrValue::String("bench".into()))
    );
}

/// `Dataset::append_staged` folds elements into a dataset staged in the same
/// session, and the provenance digest covers what is written rather than what
/// was first staged: the attributes are computed from the raw bytes, so growing
/// them without recomputing left `verify_provenance` reporting a mismatch on an
/// uncorrupted file (issue #392).
#[cfg(feature = "provenance")]
#[test]
fn appending_to_a_staged_provenance_dataset_keeps_its_hash_true() {
    use hdf5_pure::VerifyResult;

    let path = temp_path("hdf5_pure_edit_provenance_append_staged.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        let mut col = session
            .root()
            .create_dataset("col", |b| {
                b.with_f64_data(&[1.0, 2.0])
                    .with_shape(&[2])
                    .with_maxshape(&[u64::MAX])
                    .with_chunks(&[4])
                    .with_provenance("test-suite", "2026-02-19T12:00:00Z", Some("bench"));
            })
            .unwrap();
        col.append_staged(|a| {
            a.append_f64(&[3.0, 4.0]);
        })
        .unwrap();
        assert_eq!(col.shape().unwrap(), vec![4]);
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("col").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        ds.verify_provenance().unwrap(),
        VerifyResult::Ok,
        "the stored digest must cover the appended elements too"
    );
    // The rest of the provenance set is rebuilt from the same inputs, so it is
    // unchanged by the append.
    let attrs = ds.attrs().unwrap();
    assert_eq!(
        attrs.get("_provenance_creator"),
        Some(&AttrValue::String("test-suite".into()))
    );
    assert_eq!(
        attrs.get("_provenance_timestamp"),
        Some(&AttrValue::String("2026-02-19T12:00:00Z".into()))
    );
    assert_eq!(
        attrs.get("_provenance_source"),
        Some(&AttrValue::String("bench".into()))
    );
}

/// A provenance-tagged dataset can also be chunked/compressed in the same
/// commit: provenance attributes and chunked storage flow through the same
/// `attrs` vec and apply-loop path independently, so this combination should
/// just work — this test is the regression guard for that claim.
#[cfg(all(feature = "provenance", feature = "deflate"))]
#[test]
fn add_provenance_chunked_dataset_via_edit_session() {
    use hdf5_pure::VerifyResult;

    let path = temp_path("hdf5_pure_edit_add_provenance_chunked.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("sensor_chunked", |b| {
                b.with_f64_data(&[1.0, 2.0, 3.0, 4.0])
                    .with_shape(&[4])
                    .with_chunks(&[2])
                    .with_deflate(6)
                    .with_provenance("test-suite", "2026-02-19T12:00:00Z", None);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("sensor_chunked").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(ds.verify_provenance().unwrap(), VerifyResult::Ok);
    let attrs = ds.attrs().unwrap();
    assert_eq!(
        attrs.get("_provenance_creator"),
        Some(&AttrValue::String("test-suite".into()))
    );
    assert!(
        !attrs.contains_key("_provenance_source"),
        "no source was given, so no source attribute should be written"
    );
}

/// Provenance attributes (up to 4) are appended to `attrs` before the
/// dense-attribute (`MAX_COMPACT_ATTRS` = 8) budget check runs, so a dataset
/// that would otherwise fit can be pushed over the limit by its own
/// provenance metadata. Exactly at the boundary (8 total) must succeed with
/// every value intact.
#[cfg(feature = "provenance")]
#[test]
fn add_provenance_dataset_at_attr_budget_boundary_via_edit_session() {
    let path = temp_path("hdf5_pure_edit_add_provenance_at_budget.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("sensor_at_budget", |ds| {
                ds.with_f64_data(&[1.0, 2.0]);
                // 4 plain attributes + 4 provenance attributes (source included) = 8.
                for i in 0..4i64 {
                    ds.set_attr(&format!("plain_{i}"), AttrValue::I64(i));
                }
                ds.with_provenance("test-suite", "2026-02-19T12:00:00Z", Some("bench"));
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("sensor_at_budget").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0]);
    let attrs = ds.attrs().unwrap();
    for i in 0..4i64 {
        assert_eq!(
            attrs.get(&format!("plain_{i}")),
            Some(&AttrValue::I64(i)),
            "plain_{i} attribute value mismatch"
        );
    }
    assert_eq!(
        attrs.get("_provenance_creator"),
        Some(&AttrValue::String("test-suite".into()))
    );
}

/// One attribute past the boundary above (9 total, once provenance is appended)
/// no longer fits the object header, so the whole set moves to a fractal heap —
/// provenance attributes counting toward the budget exactly like the caller's own
/// (issue #102).
#[cfg(feature = "provenance")]
#[test]
fn add_provenance_dataset_over_attr_budget_uses_a_heap() {
    let path = temp_path("hdf5_pure_edit_add_provenance_over_budget.h5");
    write_starter(&path);
    assert!(!has_fractal_heap(&std::fs::read(&path).unwrap()));

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("sensor_over_budget", |ds| {
                ds.with_f64_data(&[1.0, 2.0]);
                // 5 plain attributes + 4 provenance attributes (source included) = 9.
                for i in 0..5 {
                    ds.set_attr(&format!("plain_{i}"), AttrValue::I32(i));
                }
                ds.with_provenance("test-suite", "2026-02-19T12:00:00Z", Some("bench"));
            })
            .unwrap();
        session.commit().unwrap();
    }

    assert!(
        has_fractal_heap(&std::fs::read(&path).unwrap()),
        "nine attributes are one past what the object header holds",
    );
    let file = File::open(&path).unwrap();
    let attrs = file.dataset("sensor_over_budget").unwrap().attrs().unwrap();
    assert_eq!(attrs.len(), 9);
    for i in 0..5 {
        assert_eq!(attrs.get(&format!("plain_{i}")), Some(&AttrValue::I32(i)));
    }
    assert_eq!(
        attrs.get("_provenance_creator"),
        Some(&AttrValue::String("test-suite".into())),
    );
}

/// A dataset with a variable-length attribute (issue #105) can be added in
/// place: its global heap collection is placed and its placeholder reference
/// patched during commit, alongside the dataset's own fixed-size attributes.
#[test]
fn add_dataset_with_variable_length_attribute_via_edit_session() {
    let path = temp_path("hdf5_pure_edit_add_dataset_vlen_attr.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("labeled", |ds| {
                ds.with_i32_data(&[1, 2, 3]);
                ds.set_attr(
                    "tags",
                    AttrValue::VarLenAsciiCharArray(vec![
                        "one".into(),
                        "two".into(),
                        "three".into(),
                    ]),
                );
                ds.set_attr("scale", AttrValue::F64(2.5));
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("labeled").unwrap();
    assert_eq!(ds.read_i32().unwrap(), vec![1, 2, 3]);
    let attrs = ds.attrs().unwrap();
    assert_eq!(
        attrs.get("tags"),
        Some(&AttrValue::VarLenAsciiCharArray(vec![
            "one".into(),
            "two".into(),
            "three".into()
        ]))
    );
    assert_eq!(attrs.get("scale"), Some(&AttrValue::F64(2.5)));
    // The original, pre-existing dataset is untouched.
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

/// A variable-length attribute is patched before the chunked/non-chunked
/// apply branch, so a chunked dataset can carry one too (VL attributes live
/// in the object header, not inside a chunk — only VL-string *data* is
/// refused when chunked, not VL attributes).
#[test]
fn add_chunked_dataset_with_variable_length_attribute_via_edit_session() {
    let path = temp_path("hdf5_pure_edit_add_chunked_vlen_attr.h5");
    write_starter(&path);

    let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.5).collect();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("chunky_labeled", |b| {
                b.with_f64_data(&data).with_chunks(&[25]).set_attr(
                    "tags",
                    AttrValue::VarLenAsciiCharArray(vec!["one".into(), "two".into()]),
                );
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("chunky_labeled").unwrap();
    assert_eq!(ds.read_f64().unwrap(), data);
    assert_eq!(
        ds.attrs().unwrap().get("tags"),
        Some(&AttrValue::VarLenAsciiCharArray(vec![
            "one".into(),
            "two".into()
        ]))
    );
}

/// A dataset attribute whose serialized message overflows the object header's
/// 2-byte message-size field cannot be stored compactly, so it goes to a fractal
/// heap instead of being refused (issue #102). A `VarLenAsciiCharArray` with enough
/// strings is the practical way to reach that: each element serializes to a
/// fixed-size global-heap reference, so enough of them push the message past
/// `u16::MAX` bytes.
#[test]
fn add_dataset_with_oversized_variable_length_attribute_uses_a_heap() {
    let path = temp_path("hdf5_pure_edit_add_oversized_vlen_attr.h5");
    write_starter(&path);
    assert!(!has_fractal_heap(&std::fs::read(&path).unwrap()));

    // Each element serializes to a fixed-size 16-byte global-heap reference;
    // 5000 of them (80000 bytes) comfortably overflows the object header's
    // 2-byte (`u16::MAX` = 65535) message-size field.
    let strings: Vec<String> = (0..5000).map(|i| i.to_string()).collect();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("oversized", |b| {
                b.with_i32_data(&[1])
                    .set_attr("tags", AttrValue::VarLenAsciiCharArray(strings.clone()));
            })
            .unwrap();
        session.commit().unwrap();
    }

    assert!(
        has_fractal_heap(&std::fs::read(&path).unwrap()),
        "an attribute the object header cannot describe belongs in a heap",
    );
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("oversized")
            .unwrap()
            .attrs()
            .unwrap()
            .get("tags"),
        Some(&AttrValue::VarLenAsciiCharArray(strings)),
    );
}

/// Regression test for issue #105's silent-corruption bug: a variable-length
/// string dataset (`with_vlen_strings`) added via `File::open_rw` used to commit
/// `Ok(())` without ever writing its global heap collection or patching its
/// placeholder references, so the dataset failed to read back
/// (`InvalidGlobalHeapSignature`). It must now round-trip like any other
/// added dataset.
#[test]
fn add_vlen_string_dataset_via_edit_session() {
    let path = temp_path("hdf5_pure_edit_add_vlen_string_dataset.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("labels", |b| {
                b.with_vlen_strings(&["alpha", "", "gamma"]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("labels").unwrap();
    assert_eq!(
        ds.read_string().unwrap(),
        vec!["alpha".to_string(), String::new(), "gamma".to_string()]
    );
    // The original, pre-existing dataset is untouched.
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn add_chunked_vlen_string_dataset_is_rejected_without_writing() {
    let path = temp_path("hdf5_pure_edit_add_chunked_vlen_string.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        let err = session
            .root()
            .create_dataset("labels", |b| {
                b.with_vlen_strings(&["a", "b", "c"]).with_chunks(&[2]);
            })
            .unwrap_err();
        assert!(
            err.to_string().contains("variable-length-string"),
            "got: {err}"
        );
        session.commit().unwrap();
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
}

/// An object-reference dataset (issue #105) can be added in place, targeting
/// an object that already existed before this commit — resolved via the
/// pre-commit-file fallback in `resolve_reference_target` since the target is
/// untouched by this commit.
#[test]
fn add_reference_dataset_targeting_preexisting_object_via_edit_session() {
    let path = temp_path("hdf5_pure_edit_add_ref_preexisting.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("refs", |b| {
                b.with_path_references(&["original"]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let targets = file.dataset("refs").unwrap().dereference().unwrap();
    assert_eq!(targets.len(), 1);
    match &targets[0] {
        Object::Dataset(ds) => assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]),
        other => panic!("expected a dataset reference, got {other:?}"),
    }
}

/// A reference dataset may target a sibling dataset added in the **same**
/// commit and the **same** group, regardless of which one was staged first —
/// the apply loop places every non-reference dataset in a group before any
/// reference dataset in that group (issue #105).
#[test]
fn add_reference_dataset_targeting_sibling_added_in_same_commit() {
    let path = temp_path("hdf5_pure_edit_add_ref_sibling.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        // Stage the reference dataset BEFORE its target to prove placement
        // order is independent of `pending_datasets` staging order.
        session
            .root()
            .create_dataset("refs", |b| {
                b.with_path_references(&["target"]);
            })
            .unwrap();
        session
            .root()
            .create_dataset("target", |b| {
                b.with_i32_data(&[7, 8, 9]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let targets = file.dataset("refs").unwrap().dereference().unwrap();
    assert_eq!(targets.len(), 1);
    match &targets[0] {
        Object::Dataset(ds) => assert_eq!(ds.read_i32().unwrap(), vec![7, 8, 9]),
        other => panic!("expected a dataset reference, got {other:?}"),
    }
}

/// A path with no object anywhere — neither pre-existing nor added in this
/// commit — becomes an undefined reference rather than an error, mirroring
/// the whole-file writer's resolution convention for the same builder type
/// (issue #105).
#[test]
fn add_reference_dataset_targeting_nonexistent_path_becomes_undefined() {
    let path = temp_path("hdf5_pure_edit_add_ref_nonexistent.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("refs", |b| {
                b.with_path_references(&["does/not/exist"]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("refs").unwrap();
    let err = ds.dereference().unwrap_err();
    assert!(err.to_string().contains("null/undefined"), "got: {err}");
    // The dereference error alone doesn't distinguish an undefined reference
    // (`u64::MAX`) from address 0 or any other unresolvable garbage address —
    // check the stored 8-byte element directly.
    assert_eq!(ds.read_raw().unwrap(), u64::MAX.to_le_bytes());
}

/// A reference targeting an **ancestor group of its own dataset** is refused:
/// the ancestor's own address is not known until after all of its children —
/// including this reference dataset — are placed, so resolving it now would
/// require a stale or made-up address (issue #105).
#[test]
fn add_reference_dataset_targeting_unprocessed_ancestor_is_rejected_without_writing() {
    let path = temp_path("hdf5_pure_edit_add_ref_ancestor.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("refs", |b| {
                b.with_path_references(&[""]);
            })
            .unwrap(); // root, its own parent
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("still writing"), "got: {err}");
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
}

/// A reference targeting a same-depth sibling **group** that the deepest-first
/// apply order has not reached yet is refused for the same reason as an
/// unprocessed ancestor: `"a"` sorts (and is therefore processed) before
/// `"b"`, so `"b"`'s address is not yet known when `"a/refs"` resolves
/// (issue #105).
#[test]
fn add_reference_dataset_targeting_unprocessed_sibling_group_is_rejected_without_writing() {
    let path = temp_path("hdf5_pure_edit_add_ref_sibling_group.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session.root().create_group("a").unwrap();
        session.root().create_group("b").unwrap();
        session
            .root()
            .create_dataset("a/refs", |b| {
                b.with_path_references(&["b"]);
            })
            .unwrap();
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("still writing"), "got: {err}");
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn add_chunked_reference_dataset_is_rejected_without_writing() {
    let path = temp_path("hdf5_pure_edit_add_chunked_ref.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        let err = session
            .root()
            .create_dataset("refs", |b| {
                b.with_path_references(&["original"]).with_chunks(&[1]);
            })
            .unwrap_err();
        assert!(err.to_string().contains("object-reference"), "got: {err}");
        session.commit().unwrap();
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
}

/// Every element of a multi-element `with_path_references` array is resolved
/// independently and in order — repeating a target (element 0 and 2 both
/// point at `original`) also catches an indexing bug that a single-target
/// test cannot.
#[test]
fn add_reference_dataset_with_multiple_elements_via_edit_session() {
    let path = temp_path("hdf5_pure_edit_add_ref_multi_element.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("second", |b| {
                b.with_i32_data(&[42]);
            })
            .unwrap();
        session
            .root()
            .create_dataset("refs", |b| {
                b.with_path_references(&["original", "second", "original"]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let targets = file.dataset("refs").unwrap().dereference().unwrap();
    assert_eq!(targets.len(), 3);
    match &targets[0] {
        Object::Dataset(ds) => assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]),
        other => panic!("expected a dataset reference, got {other:?}"),
    }
    match &targets[1] {
        Object::Dataset(ds) => assert_eq!(ds.read_i32().unwrap(), vec![42]),
        other => panic!("expected a dataset reference, got {other:?}"),
    }
    match &targets[2] {
        Object::Dataset(ds) => assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]),
        other => panic!("expected a dataset reference, got {other:?}"),
    }
}

/// A reference can resolve to a **group**, not just a dataset — the existing
/// positive reference tests only ever target a dataset, so `Object::Group`'s
/// arm of `dereference()`'s result was previously unexercised.
#[test]
fn add_reference_dataset_targeting_a_group_via_edit_session() {
    let path = temp_path("hdf5_pure_edit_add_ref_group.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("original")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0]);
    let mut g = b.create_group("grp");
    g.create_dataset("inner").with_i32_data(&[5, 6]);
    g.set_attr("tag", AttrValue::I64(7));
    b.add_group(g.finish());
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("refs", |b| {
                b.with_path_references(&["grp"]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let targets = file.dataset("refs").unwrap().dereference().unwrap();
    assert_eq!(targets.len(), 1);
    match &targets[0] {
        Object::Group(g) => {
            assert_eq!(g.dataset("inner").unwrap().read_i32().unwrap(), vec![5, 6]);
            assert_eq!(g.attrs().unwrap().get("tag"), Some(&AttrValue::I64(7)));
        }
        other => panic!("expected a group reference, got {other:?}"),
    }
}

/// `with_path_references(&[])` is a valid degenerate case, consistent with
/// the empty-dataset and empty-vlen-string-dataset edge cases already
/// supported elsewhere in this #105 stack: no target needs resolving, and the
/// layout falls back to the same undefined-address sentinel as any other
/// empty dataset.
#[test]
fn add_zero_element_reference_dataset_via_edit_session() {
    let path = temp_path("hdf5_pure_edit_add_ref_zero_element.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("refs", |b| {
                b.with_path_references(&[]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("refs").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![0]);
    assert_eq!(ds.dereference().unwrap().len(), 0);
}

/// Regression test: a reference targeting an object **deleted in the same
/// commit** must not resolve to that object's soon-to-be-freed pre-commit
/// address — `resolve_reference_target` must check `pending_deletes`, not
/// just `nodes`/`add_targets`/`write_targets`. A nested group/dataset is
/// staged first so it would already be durably written (deepest-first apply
/// order) before the root-level delete is even reached; the whole commit must
/// still leave the file untouched.
#[test]
fn add_reference_dataset_targeting_same_commit_delete_is_rejected_without_writing() {
    let path = temp_path("hdf5_pure_edit_add_ref_deleted_target.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session.root().create_group("grp").unwrap();
        session
            .root()
            .create_dataset("grp/inner", |b| {
                b.with_i32_data(&[1, 2, 3]);
            })
            .unwrap();
        session.root().delete("original").unwrap();
        session
            .root()
            .create_dataset("refs", |b| {
                b.with_path_references(&["original"]);
            })
            .unwrap();
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string().contains("this commit deletes"),
            "got: {err}"
        );
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
}

/// A deletion takes the whole subtree with it, so a reference to a *child* of
/// a deleted group dangles exactly as a reference to the group itself does —
/// and that one was refused while the child's was not, because the guard
/// matched the deleted path exactly instead of by prefix (issue #314).
///
/// The committed reference pointed at the freed header: measured on the
/// commit before the fix, the stored address landed inside a span the same
/// commit reported as reusable free space, and one later commit that reused
/// it turned a clean `dereference` into `InvalidObjectHeaderVersion(170)` —
/// the filler byte, read as an object header.
///
/// Both depths are checked. A direct child catches the exact-match guard the
/// bug shipped; a grandchild catches a fix that only descends one level.
#[test]
fn add_reference_dataset_targeting_a_child_of_a_same_commit_delete_is_rejected() {
    let path = temp_path("hdf5_pure_edit_add_ref_deleted_child.h5");
    let mut b = FileBuilder::new();
    let mut doomed = b.create_group("doomed");
    doomed.create_dataset("inner").with_i32_data(&[7, 7, 7]);
    let mut sub = doomed.create_group("sub");
    sub.create_dataset("deep").with_i32_data(&[8]);
    doomed.add_group(sub.finish());
    b.add_group(doomed.finish());
    b.write(&path).unwrap();
    // One fixture for both depths: each refusal is asserted to leave the file
    // byte-identical, so the second pass runs against a file that has already
    // survived a failed commit.
    let before = std::fs::read(&path).unwrap();

    for target in ["doomed/inner", "doomed/sub/deep"] {
        {
            let session = File::open_rw(&path).unwrap();
            session.root().delete("doomed").unwrap();
            session
                .root()
                .create_dataset("refs", |b| {
                    b.with_path_references(&[target]);
                })
                .unwrap();
            let err = session.commit().unwrap_err();
            // Assert the delete guard's own message: every other guard in
            // `resolve_reference_target` reports "still writing", so a test
            // that only asserted `is_err` would pass on any of them.
            assert!(
                err.to_string().contains("this commit deletes"),
                "{target}: got: {err}"
            );
        }

        assert_eq!(std::fs::read(&path).unwrap(), before, "{target}");
    }
}

/// The floor under the guard above: a delete in the commit must not stop an
/// unrelated reference from resolving. Without this, a guard that refused
/// whenever *any* deletion was staged would pass every other reference test
/// in this file — none of them combines a delete with a reference that is
/// supposed to succeed.
#[test]
fn a_delete_elsewhere_does_not_block_an_unrelated_reference() {
    let path = temp_path("hdf5_pure_edit_add_ref_delete_elsewhere.h5");
    let mut b = FileBuilder::new();
    let mut doomed = b.create_group("doomed");
    doomed.create_dataset("inner").with_i32_data(&[7]);
    b.add_group(doomed.finish());
    let mut keep = b.create_group("keep");
    keep.create_dataset("survivor").with_i32_data(&[4, 5, 6]);
    b.add_group(keep.finish());
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("doomed").unwrap();
        session
            .root()
            .create_dataset("refs", |b| {
                b.with_path_references(&["keep/survivor"]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let targets = file.dataset("refs").unwrap().dereference().unwrap();
    assert_eq!(targets.len(), 1);
    match &targets[0] {
        Object::Dataset(ds) => assert_eq!(ds.read_i32().unwrap(), vec![4, 5, 6]),
        other => panic!("expected a dataset reference, got {other:?}"),
    }
    drop(file);
}

/// The interaction with issue #305: when the commit puts the deleted path
/// *back*, a reference to the replacement's child resolves to the **new**
/// object. The deepest-first apply order places `g/inner` before the root
/// group that references it, so step 1 (`path_addr`) answers and no guard is
/// consulted.
///
/// That is also this test's limitation, and it is worth stating rather than
/// discovering: because the guard is never reached, no mutation of the delete
/// test makes this fail. It backs the documented positive claim and asserts
/// the resolved *value* — the replacement, not the object removed — which no
/// other test here does. It is not evidence that the guard is correct.
#[test]
fn a_reference_to_a_replaced_paths_new_child_resolves_to_the_replacement() {
    let path = temp_path("hdf5_pure_edit_add_ref_replaced_child.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("g");
    g.create_dataset("inner").with_i32_data(&[7, 7, 7]);
    b.add_group(g.finish());
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("g").unwrap();
        session.root().create_group("g").unwrap();
        session
            .root()
            .create_dataset("g/inner", |b| {
                b.with_i32_data(&[9, 9]);
            })
            .unwrap();
        session
            .root()
            .create_dataset("refs", |b| {
                b.with_path_references(&["g/inner"]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let targets = file.dataset("refs").unwrap().dereference().unwrap();
    assert_eq!(targets.len(), 1);
    match &targets[0] {
        // The replacement, not the [7, 7, 7] the commit removed.
        Object::Dataset(ds) => assert_eq!(ds.read_i32().unwrap(), vec![9, 9]),
        other => panic!("expected a dataset reference, got {other:?}"),
    }
    drop(file);
}

/// The delete guard runs *after* the three "still writing" ones, so a
/// replacement the apply loop has merely not reached yet is reported as the
/// ordering problem it is rather than as a deletion. `a` and `b` are both at
/// depth 1 and `a` sorts first, so `a/refs` resolves `b/inner` before `b` is
/// placed — even though `b` is deleted and recreated in this same commit, the
/// object being referenced is one the commit *writes*, and saying it is being
/// deleted would send a reader looking for the wrong thing.
///
/// Both orderings refuse, so this pins a diagnostic rather than a correctness
/// property. It is here because the ordering is a deliberate choice that
/// nothing else in the suite would notice being undone.
///
/// The ordering is a better default, not a partition. A child of a replaced
/// path that the replacement does *not* recreate is genuinely doomed and also
/// reports "still writing", because `add_targets` claims the whole replaced
/// subtree by prefix. That case is deliberately left untested: the message it
/// gets is the wrong one, and a test asserting it would make the imprecision
/// harder to fix rather than easier.
#[test]
fn a_reference_to_an_unplaced_replacement_reports_the_ordering_not_the_delete() {
    let path = temp_path("hdf5_pure_edit_add_ref_unplaced_replacement.h5");
    let mut b = FileBuilder::new();
    let mut a = b.create_group("a");
    a.create_dataset("seed").with_i32_data(&[0]);
    b.add_group(a.finish());
    let mut later = b.create_group("b");
    later.create_dataset("inner").with_i32_data(&[7]);
    b.add_group(later.finish());
    b.write(&path).unwrap();

    let session = File::open_rw(&path).unwrap();
    session.root().delete("b").unwrap();
    session.root().create_group("b").unwrap();
    session
        .root()
        .create_dataset("b/inner", |b| {
            b.with_i32_data(&[9]);
        })
        .unwrap();
    session
        .root()
        .create_dataset("a/refs", |b| {
            b.with_path_references(&["b/inner"]);
        })
        .unwrap();
    let err = session.commit().unwrap_err();
    assert!(err.to_string().contains("still writing"), "got: {err}");
    drop(session);
}

/// Regression test: a reference targeting a **copy destination** in the same
/// commit is refused (a copy's root address is never recorded in
/// `path_addr`), and — like the delete case above — an earlier-processed
/// nested addition must not leak into the file when the refusal fires later.
#[test]
fn add_reference_dataset_targeting_copy_destination_is_rejected_without_writing() {
    let path = temp_path("hdf5_pure_edit_add_ref_copy_dest.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session.root().create_group("grp").unwrap();
        session
            .root()
            .create_dataset("grp/inner", |b| {
                b.with_i32_data(&[1, 2, 3]);
            })
            .unwrap();
        session.copy("original", "dup").unwrap();
        session
            .root()
            .create_dataset("refs", |b| {
                b.with_path_references(&["dup"]);
            })
            .unwrap();
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("still writing"), "got: {err}");
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
}

/// Regression test: a reference targeting a `write_dataset` (value-overwrite)
/// target in the same commit is refused, conservatively, regardless of
/// whether the overwrite would relocate the header or land in place — and,
/// again, an earlier-processed nested addition must not leak.
#[test]
fn add_reference_dataset_targeting_write_overwrite_target_is_rejected_without_writing() {
    let path = temp_path("hdf5_pure_edit_add_ref_write_target.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session.root().create_group("grp").unwrap();
        session
            .root()
            .create_dataset("grp/inner", |b| {
                b.with_i32_data(&[1, 2, 3]);
            })
            .unwrap();
        session
            .dataset("original")
            .unwrap()
            .write_staged(|b| {
                b.with_f64_data(&[9.0, 9.0, 9.0, 9.0]);
            })
            .unwrap();
        session
            .root()
            .create_dataset("refs", |b| {
                b.with_path_references(&["original"]);
            })
            .unwrap();
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("still writing"), "got: {err}");
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
}

/// The fixture the address-side reference tests below share (issue #317): a
/// dataset inside a group a commit can delete, and a reference dataset naming
/// it by path. Returns the address `refs` stores — which is what a caller hands
/// to `with_reference_data` to stage the *same* reference as an address rather
/// than as a path, the form `resolve_reference_target` never sees.
///
/// `inner` is deliberately large enough that the group's reclaimed span is wide
/// and unmistakable; the assertions below do not depend on its contents.
fn write_reference_fixture(path: &std::path::Path) -> u64 {
    let mut b = FileBuilder::new();
    let mut g = b.create_group("g");
    g.create_dataset("inner")
        .with_i32_data(&(0..512).collect::<Vec<i32>>());
    b.add_group(g.finish());
    b.create_dataset("refs").with_path_references(&["g/inner"]);
    b.write(path).unwrap();

    let file = File::open(path).unwrap();
    let raw = file.dataset("refs").unwrap().read_raw().unwrap();
    u64::from_le_bytes(raw[..8].try_into().unwrap())
}

/// A reference *added* as a resolved address, naming an object the same commit
/// deletes, is refused rather than written (issue #317).
///
/// `with_reference_data` stages element bytes that are already addresses, so it
/// sets no `reference_targets` and never reaches `resolve_reference_target` —
/// the function carrying the by-name delete refusal that the identical target
/// gets when it is named as a path (issue #314). Before the screen this commit
/// returned `Ok`, leaving `added` pointing into the span the same commit had
/// just reported as reusable.
#[test]
fn an_added_reference_address_into_deleted_space_is_refused() {
    let path = temp_path("hdf5_pure_edit_ref_addr_add.h5");
    let inner = write_reference_fixture(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("added", |b| {
                b.with_reference_data(&[inner]);
            })
            .unwrap();
        session.root().delete("g").unwrap();
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string()
                .contains("holds the address of an object this commit deletes"),
            "got: {err}"
        );
        // A refusal returns the whole batch to the caller (issue #316).
        assert!(session.has_staged_edits());
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
}

/// The same address through the *overwrite* door, which reaches neither
/// `resolve_reference_target` nor `preflight_reference_targets`: a staged write
/// replaces an existing dataset's bytes instead of placing a new object, so it
/// is screened on its own (issue #317).
#[test]
fn an_overwritten_reference_address_into_deleted_space_is_refused() {
    let path = temp_path("hdf5_pure_edit_ref_addr_write.h5");
    let inner = write_reference_fixture(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("refs")
            .unwrap()
            .write_staged(|b| {
                b.with_reference_data(&[inner]);
            })
            .unwrap();
        session.root().delete("g").unwrap();
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string()
                .contains("holds the address of an object this commit deletes"),
            "got: {err}"
        );
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
}

/// An address naming an object the delete does not touch still commits, and
/// dereferences to it (issue #317).
///
/// This is what makes the screen a test of *where the address points* rather
/// than of whether the commit deletes anything: `doomed` goes away in the same
/// commit that writes a reference to `keep/survivor`, and only the second fact
/// decides the verdict.
#[test]
fn a_reference_address_to_a_surviving_object_is_accepted_beside_a_delete() {
    let path = temp_path("hdf5_pure_edit_ref_addr_survivor.h5");
    let mut b = FileBuilder::new();
    let mut doomed = b.create_group("doomed");
    doomed.create_dataset("inner").with_i32_data(&[7]);
    b.add_group(doomed.finish());
    let mut keep = b.create_group("keep");
    keep.create_dataset("survivor").with_i32_data(&[4, 5, 6]);
    b.add_group(keep.finish());
    b.create_dataset("refs")
        .with_path_references(&["keep/survivor"]);
    b.write(&path).unwrap();

    let survivor = {
        let file = File::open(&path).unwrap();
        let raw = file.dataset("refs").unwrap().read_raw().unwrap();
        u64::from_le_bytes(raw[..8].try_into().unwrap())
    };

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("added", |b| {
                b.with_reference_data(&[survivor]);
            })
            .unwrap();
        session.root().delete("doomed").unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let targets = file.dataset("added").unwrap().dereference().unwrap();
    assert_eq!(targets.len(), 1);
    match &targets[0] {
        Object::Dataset(ds) => assert_eq!(ds.read_i32().unwrap(), vec![4, 5, 6]),
        other => panic!("expected a dataset reference, got {other:?}"),
    }
    drop(file);
}

/// An in-file copy re-emits its source's element bytes verbatim, which for an
/// object-reference dataset means re-emitting addresses. The copy of `refs`
/// would land pointing at `g/inner`, which the same commit deletes, so the
/// copied elements are screened exactly as a staged dataset's are and the
/// commit is refused (issue #317).
#[test]
fn a_copy_of_a_reference_dataset_beside_a_delete_is_refused() {
    let path = temp_path("hdf5_pure_edit_ref_copy_delete.h5");
    write_reference_fixture(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session.copy("refs", "refs_copy").unwrap();
        session.root().delete("g").unwrap();
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string()
                .contains("holds the address of an object this commit deletes"),
            "got: {err}"
        );
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
}

/// The other half of that rule: with nothing deleted there is no reclaimed
/// space to point into, so the same copy commits and both datasets dereference
/// to the same object (issue #317).
///
/// The companion to `copy_same_file_still_allows_variable_length_attribute`:
/// an in-file copy keeps addresses valid by sharing the file, and only a
/// removal in the same commit takes that away.
#[test]
fn a_copy_of_a_reference_dataset_without_a_delete_still_works() {
    let path = temp_path("hdf5_pure_edit_ref_copy_plain.h5");
    write_reference_fixture(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session.copy("refs", "refs_copy").unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    for name in ["refs", "refs_copy"] {
        let targets = file.dataset(name).unwrap().dereference().unwrap();
        assert_eq!(targets.len(), 1, "{name}");
        match &targets[0] {
            Object::Dataset(ds) => {
                assert_eq!(ds.read_i32().unwrap().len(), 512, "{name}");
            }
            other => panic!("{name}: expected a dataset reference, got {other:?}"),
        }
    }
    drop(file);
}

/// A copy of a reference dataset commits *beside* a delete when the reference
/// names something the delete does not take: `refs` points at `g/inner`, and
/// `scratch` is what goes away (issue #317).
///
/// This is what makes the copy screen a test of where the copied addresses
/// point rather than of whether the commit deletes anything. The distinction is
/// not academic: every successful delete reclaims at least the deleted object's
/// own header, so a screen gated on "this commit deletes something" would
/// refuse every copy of a reference dataset in any commit that removes
/// anything at all.
#[test]
fn a_copy_of_a_reference_dataset_is_allowed_beside_an_unrelated_delete() {
    let path = temp_path("hdf5_pure_edit_ref_copy_unrelated.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("g");
    g.create_dataset("inner").with_i32_data(&[1, 2, 3]);
    b.add_group(g.finish());
    b.create_dataset("refs").with_path_references(&["g/inner"]);
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session.root().create_group("scratch").unwrap();
        session.commit().unwrap();
        session.copy("refs", "refs_copy").unwrap();
        session.root().delete("scratch").unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let targets = file.dataset("refs_copy").unwrap().dereference().unwrap();
    match &targets[0] {
        Object::Dataset(ds) => assert_eq!(ds.read_i32().unwrap(), vec![1, 2, 3]),
        other => panic!("expected a dataset reference, got {other:?}"),
    }
    drop(file);
}

/// An address naming an object the same commit *rewrites elsewhere* is refused
/// too: the object still exists, but not there (issue #317).
///
/// Adding a child to `g` rebuilds its header at a fresh address and frees the
/// old one, so a reference supplied as `g`'s pre-commit address is stale the
/// moment the commit lands — and, once something reuses the span, reads as
/// whatever went there. The same target named as a **path** resolves to the new
/// address instead — placed already, because `commit` writes the deepest groups
/// first — which is what the refusal points at. It does not always work, and the
/// relocated-dataset test below pins the case where it does not.
#[test]
fn a_reference_address_to_a_group_this_commit_rewrites_is_refused() {
    let path = temp_path("hdf5_pure_edit_ref_addr_moved_group.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("g");
    g.create_dataset("inner").with_i32_data(&[1, 2, 3]);
    b.add_group(g.finish());
    b.create_dataset("refs").with_path_references(&["g"]);
    b.write(&path).unwrap();

    let g_addr = {
        let file = File::open(&path).unwrap();
        let raw = file.dataset("refs").unwrap().read_raw().unwrap();
        u64::from_le_bytes(raw[..8].try_into().unwrap())
    };
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("added", |b| {
                b.with_reference_data(&[g_addr]);
            })
            .unwrap();
        session
            .root()
            .create_dataset("g/newthing", |b| {
                b.with_i32_data(&[9]);
            })
            .unwrap();
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("rewrites elsewhere"), "got: {err}");
    }
    assert_eq!(std::fs::read(&path).unwrap(), before);

    // The same commit, with the target named by path, commits — and the
    // reference resolves to the rebuilt group, `newthing` included.
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("added", |b| {
                b.with_path_references(&["g"]);
            })
            .unwrap();
        session
            .root()
            .create_dataset("g/newthing", |b| {
                b.with_i32_data(&[9]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    match &file.dataset("added").unwrap().dereference().unwrap()[0] {
        Object::Group(g) => {
            let mut names = g.datasets().unwrap();
            names.sort();
            assert_eq!(names, vec!["inner".to_string(), "newthing".to_string()]);
        }
        other => panic!("expected a group reference, got {other:?}"),
    }
    drop(file);
}

/// The same rule for a *dataset* target, relocated by an attribute edit rather
/// than by gaining a child (issue #317).
///
/// An attribute edit rewrites the dataset's object header at a fresh address —
/// its data stays put — so a reference holding the old header address is left
/// naming freed bytes.
#[test]
fn a_reference_address_to_a_relocated_dataset_is_refused() {
    let path = temp_path("hdf5_pure_edit_ref_addr_moved_dataset.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[11, 22, 33]);
    b.create_dataset("refs").with_path_references(&["d"]);
    b.write(&path).unwrap();

    let d_addr = {
        let file = File::open(&path).unwrap();
        let raw = file.dataset("refs").unwrap().read_raw().unwrap();
        u64::from_le_bytes(raw[..8].try_into().unwrap())
    };
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("added", |b| {
                b.with_reference_data(&[d_addr]);
            })
            .unwrap();
        session
            .dataset("d")
            .unwrap()
            .set_attr("tag", AttrValue::I32(1))
            .unwrap();
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("rewrites elsewhere"), "got: {err}");
    }

    // The refusal's first suggestion does not reach this one: a written dataset
    // is a `write_targets` entry, which the path side refuses outright rather
    // than resolving. That is why the message names separate commits as well.
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("added", |b| {
                b.with_path_references(&["d"]);
            })
            .unwrap();
        session
            .dataset("d")
            .unwrap()
            .set_attr("tag", AttrValue::I32(1))
            .unwrap();
        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("still writing"), "got: {err}");
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
}

/// A datatype mixing an object reference this crate can locate with one it
/// cannot is refused, not screened half-way (issue #317).
///
/// This is the shape that defeats a screen keyed on "did the walker find any
/// slots": the 8-byte member yields one, so the list is non-empty, and a screen
/// that took that for "this type is walkable" would write the other member's
/// address unexamined. `embedded_reference_slots` reports the whole datatype as
/// unaddressable instead, which is what its fall-through arm asks
/// `datatype_holds_object_reference` in order to know.
///
/// The variable-length half is what the reference C library writes (`H5T_VLEN`
/// of `H5T_STD_REF_OBJ`), so a plain `copy` reaches it; the 16-byte reference is
/// this crate's own construction. Both arrive through `with_raw_data`, which
/// takes whatever `Datatype` it is given.
#[test]
fn a_datatype_mixing_locatable_and_unlocatable_references_is_refused() {
    let object_ref = || Datatype::Reference {
        size: 8,
        ref_type: ReferenceType::Object,
    };
    let unlocatable = [
        (
            "wider than eight bytes",
            Datatype::Reference {
                size: 16,
                ref_type: ReferenceType::Object,
            },
        ),
        (
            "a variable length of them",
            Datatype::VariableLength {
                is_string: false,
                padding: None,
                charset: None,
                base_type: Box::new(object_ref()),
            },
        ),
    ];

    for (tag, second) in unlocatable {
        let path = temp_path("hdf5_pure_edit_ref_mixed.h5");
        let inner = write_reference_fixture(&path);
        let before = std::fs::read(&path).unwrap();

        let dt = CompoundTypeBuilder::with_size(24)
            .field("locatable", 0, object_ref())
            .field("other", 8, second)
            .build()
            .unwrap();
        // The hazardous address goes in the member the walker cannot reach; the
        // one it can reach is left null, so only the unreachable half can refuse.
        let mut element = vec![0u8; 24];
        element[8..16].copy_from_slice(&inner.to_le_bytes());

        {
            let session = File::open_rw(&path).unwrap();
            session
                .root()
                .create_dataset("added", |b| {
                    b.with_raw_data(dt.clone(), element.clone(), 1);
                })
                .unwrap();
            session.root().delete("g").unwrap();
            let err = session.commit().unwrap_err();
            assert!(
                err.to_string().contains("this screen cannot read"),
                "{tag}: got {err}"
            );
        }

        assert_eq!(std::fs::read(&path).unwrap(), before, "{tag}");
    }
}

/// A **dataset-region** reference is refused beside a delete, like every other
/// reference whose address this screen cannot read (issue #317).
///
/// Its element bytes are a global-heap id, and the object address sits in the
/// heap object that id names — one indirection further out than the element
/// bytes this screen walks. So it is unreadable rather than absent, which is why
/// `datatype_holds_object_address` counts it and the walker declines to map it.
/// Nothing in this crate builds one; `with_raw_data` and an in-file `copy` of a
/// C-written file are the doors.
#[test]
fn a_dataset_region_reference_is_refused_beside_a_delete() {
    let path = temp_path("hdf5_pure_edit_ref_region.h5");
    let inner = write_reference_fixture(&path);
    let before = std::fs::read(&path).unwrap();

    let mut element = vec![0u8; 12];
    element[..8].copy_from_slice(&inner.to_le_bytes());

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("added", |b| {
                b.with_raw_data(
                    Datatype::Reference {
                        size: 12,
                        ref_type: ReferenceType::DatasetRegion,
                    },
                    element.clone(),
                    1,
                );
            })
            .unwrap();
        session.root().delete("g").unwrap();
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string().contains("this screen cannot read"),
            "got: {err}"
        );
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
}

/// The relocating-write half of `moved`, through the two doors an attribute
/// edit does not cover: a **chunked rebuild** (a filtered chunk whose
/// replacement no longer fits its slot — the element count is unchanged, so
/// this is not a reshape) and a **staged append**, each of which rewrites the
/// dataset's header at a fresh address (issue #317).
///
/// Both reach `moved` from the write plan rather than from the path, so a
/// reference supplied as the target's pre-commit address is refused where the
/// same commit relocates it.
#[test]
fn a_reference_address_to_a_dataset_a_write_relocates_is_refused() {
    for (tag, relocate) in [
        (
            // A filtered chunk whose replacement compresses worse than what it
            // replaces no longer fits its slot, so the dataset is rebuilt
            // elsewhere and its old header vacated.
            "refiltered",
            (|session: &File| {
                let incompressible: Vec<i32> = (0..64)
                    .map(|i: i32| i.wrapping_mul(0x9E37_79B1u32 as i32))
                    .collect();
                session
                    .dataset("d")
                    .unwrap()
                    .write_staged(|b| {
                        b.with_i32_data(&incompressible);
                    })
                    .unwrap();
            }) as fn(&File),
        ),
        (
            "append",
            (|session: &File| {
                session
                    .dataset("d")
                    .unwrap()
                    .append_staged(|a| {
                        a.append_i32(&[9]);
                    })
                    .unwrap();
            }) as fn(&File),
        ),
    ] {
        let path = temp_path(&format!("hdf5_pure_edit_ref_moved_{tag}.h5"));
        let mut b = FileBuilder::new();
        // Zeros so every chunk compresses to almost nothing, leaving slots the
        // replacement below cannot fit back into.
        b.create_dataset("d")
            .with_i32_data(&vec![0i32; 64])
            .with_shape(&[64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[16])
            .with_deflate(6);
        b.create_dataset("refs").with_path_references(&["d"]);
        b.write(&path).unwrap();

        let d_addr = {
            let file = File::open(&path).unwrap();
            let raw = file.dataset("refs").unwrap().read_raw().unwrap();
            u64::from_le_bytes(raw[..8].try_into().unwrap())
        };
        let before = std::fs::read(&path).unwrap();

        {
            let session = File::open_rw(&path).unwrap();
            session
                .root()
                .create_dataset("added", |b| {
                    b.with_reference_data(&[d_addr]);
                })
                .unwrap();
            relocate(&session);
            let err = session.commit().unwrap_err();
            assert!(
                err.to_string().contains("rewrites elsewhere"),
                "{tag}: got {err}"
            );
        }

        assert_eq!(std::fs::read(&path).unwrap(), before, "{tag}");
    }
}

/// An address naming an object the commit leaves alone still commits, even
/// though the commit rewrites *something* — it rebuilds the root group, as every
/// commit does (issue #317).
///
/// This is what keeps the moved-header half a screen rather than a ban: every
/// commit that reaches the screen has rebuilt its root group, so the list of
/// rewritten headers is never empty, and a gate on "does this commit rewrite
/// anything" would refuse every supplied address there is.
#[test]
fn a_reference_address_to_an_object_this_commit_leaves_alone_is_accepted() {
    let path = temp_path("hdf5_pure_edit_ref_addr_untouched.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[11, 22, 33]);
    b.create_dataset("refs").with_path_references(&["d"]);
    b.write(&path).unwrap();

    let d_addr = {
        let file = File::open(&path).unwrap();
        let raw = file.dataset("refs").unwrap().read_raw().unwrap();
        u64::from_le_bytes(raw[..8].try_into().unwrap())
    };

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("added", |b| {
                b.with_reference_data(&[d_addr]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    match &file.dataset("added").unwrap().dereference().unwrap()[0] {
        Object::Dataset(ds) => assert_eq!(ds.read_i32().unwrap(), vec![11, 22, 33]),
        other => panic!("expected a dataset reference, got {other:?}"),
    }
    drop(file);
}

/// The screen reads every element, not just the first: here element 0 names a
/// survivor and element 1 names the deleted object (issue #317).
#[test]
fn a_hazardous_reference_past_the_first_element_is_found() {
    let path = temp_path("hdf5_pure_edit_ref_addr_second_element.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("g");
    g.create_dataset("inner").with_i32_data(&[1, 2, 3]);
    b.add_group(g.finish());
    let mut keep = b.create_group("keep");
    keep.create_dataset("survivor").with_i32_data(&[4, 5, 6]);
    b.add_group(keep.finish());
    b.create_dataset("probe")
        .with_path_references(&["keep/survivor", "g/inner"]);
    b.write(&path).unwrap();

    let (survivor, inner) = {
        let file = File::open(&path).unwrap();
        let raw = file.dataset("probe").unwrap().read_raw().unwrap();
        (
            u64::from_le_bytes(raw[..8].try_into().unwrap()),
            u64::from_le_bytes(raw[8..16].try_into().unwrap()),
        )
    };
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("added", |b| {
                b.with_reference_data(&[survivor, inner]);
            })
            .unwrap();
        session.root().delete("g").unwrap();
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string()
                .contains("holds the address of an object this commit deletes"),
            "got: {err}"
        );
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
}

/// A copied *group* is screened through its whole subtree, not just at its root:
/// the reference dataset here is a child of the group being copied (issue #317).
#[test]
fn a_copied_groups_subtree_is_screened() {
    let path = temp_path("hdf5_pure_edit_ref_copy_subtree.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("g");
    g.create_dataset("inner").with_i32_data(&[1, 2, 3]);
    b.add_group(g.finish());
    let mut holder = b.create_group("holder");
    holder
        .create_dataset("refs")
        .with_path_references(&["g/inner"]);
    b.add_group(holder.finish());
    b.write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session.copy("holder", "holder_copy").unwrap();
        session.root().delete("g").unwrap();
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string()
                .contains("holds the address of an object this commit deletes"),
            "got: {err}"
        );
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
}

/// Build a file with a *chunked* object-reference dataset `crefs` holding two
/// references to `keep/survivor`, alongside the `probe` path-reference dataset
/// the address is read back from and a deletable group `g`. Returns the address
/// `keep/survivor` sits at.
///
/// Two passes because the address is only known once the layout is final: the
/// second writes the same objects in the same order and changes eight data
/// bytes per element, so it must not move anything, and the caller asserts that
/// it did not.
fn write_chunked_reference_file(path: &std::path::Path, stored: u64) -> u64 {
    let mut b = FileBuilder::new();
    let mut g = b.create_group("g");
    g.create_dataset("inner").with_i32_data(&[1, 2, 3]);
    b.add_group(g.finish());
    let mut keep = b.create_group("keep");
    keep.create_dataset("survivor").with_i32_data(&[4, 5, 6]);
    b.add_group(keep.finish());
    b.create_dataset("probe")
        .with_path_references(&["keep/survivor"]);
    b.create_dataset("crefs")
        .with_reference_data(&[stored, stored])
        .with_chunks(&[1]);
    b.write(path).unwrap();

    let file = File::open(path).unwrap();
    let raw = file.dataset("probe").unwrap().read_raw().unwrap();
    u64::from_le_bytes(raw[..8].try_into().unwrap())
}

/// A copy of a *chunked* object-reference dataset is refused beside a delete
/// even though the address it holds is fine, because this path cannot tell:
/// the copy carries `chunk_bytes` exactly as the source stored them, filters
/// and all, and never decodes them (issue #317).
///
/// `crefs` names `keep/survivor` and the delete takes `g`, so the refusal can
/// only be coming from the datatype. This is the one place the screen is
/// conservative, and it is the same limit that makes `repack` refuse a chunked
/// object-reference dataset outright.
#[test]
fn a_chunked_reference_copy_beside_a_delete_is_refused() {
    let path = temp_path("hdf5_pure_edit_ref_copy_chunked.h5");
    let survivor = write_chunked_reference_file(&path, 0);
    assert_eq!(
        write_chunked_reference_file(&path, survivor),
        survivor,
        "the second pass must not move keep/survivor"
    );
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session.copy("crefs", "dup").unwrap();
        session.root().delete("g").unwrap();
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string().contains("chunked object-reference dataset"),
            "got: {err}"
        );
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
}

/// The chunked refusal above is gated on the commit reclaiming something: with
/// nothing deleted, a chunked object-reference dataset copies like any other
/// and both copies still dereference (issue #317).
///
/// Without that gate the screen would take a capability away from every commit
/// rather than from the ones where an address could have gone stale.
#[test]
fn a_chunked_reference_copy_without_a_delete_still_works() {
    let path = temp_path("hdf5_pure_edit_ref_copy_chunked_ok.h5");
    let survivor = write_chunked_reference_file(&path, 0);
    assert_eq!(write_chunked_reference_file(&path, survivor), survivor);

    {
        let session = File::open_rw(&path).unwrap();
        session.copy("crefs", "dup").unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    for name in ["crefs", "dup"] {
        let targets = file.dataset(name).unwrap().dereference().unwrap();
        assert_eq!(targets.len(), 2, "{name}");
        match &targets[0] {
            Object::Dataset(ds) => assert_eq!(ds.read_i32().unwrap(), vec![4, 5, 6], "{name}"),
            other => panic!("{name}: expected a dataset reference, got {other:?}"),
        }
    }
    drop(file);
}

/// Build a file whose `crefs` dataset holds one object reference to `g/inner`
/// through a **committed** (shared) datatype, and returns that address.
///
/// Two passes because the address is only known once the layout is final: the
/// second writes the same objects in the same order and changes eight data
/// bytes, so it must not move anything, and the caller asserts that it did not.
fn write_committed_reference_file(path: &std::path::Path, stored: u64) -> u64 {
    let mut b = FileBuilder::new();
    b.commit_datatype(
        "reftype",
        Datatype::Reference {
            size: 8,
            ref_type: ReferenceType::Object,
        },
    );
    let mut g = b.create_group("g");
    g.create_dataset("inner").with_i32_data(&[1, 2, 3]);
    b.add_group(g.finish());
    b.create_dataset("probe").with_path_references(&["g/inner"]);
    b.create_dataset("crefs")
        .with_reference_data(&[stored])
        .with_committed_datatype("reftype");
    b.write(path).unwrap();

    let file = File::open(path).unwrap();
    let raw = file.dataset("probe").unwrap().read_raw().unwrap();
    u64::from_le_bytes(raw[..8].try_into().unwrap())
}

/// A committed (shared) datatype is *resolved* before the copy's elements are
/// screened, so a copy whose named type turns out to be an object reference is
/// caught by address like any other (issue #317).
///
/// Without the resolution step there is nothing to parse at a shared Datatype
/// message — its body is a pointer into the file's shared-message storage — and
/// this copy would go through unscreened.
#[test]
fn a_copy_of_a_committed_reference_datatype_is_screened() {
    let path = temp_path("hdf5_pure_edit_ref_copy_committed.h5");
    let inner = write_committed_reference_file(&path, 0);
    assert_eq!(
        write_committed_reference_file(&path, inner),
        inner,
        "the second pass must not move g/inner"
    );
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session.copy("crefs", "dup").unwrap();
        session.root().delete("g").unwrap();
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string()
                .contains("holds the address of an object this commit deletes"),
            "got: {err}"
        );
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
}

/// The other side of that resolution: a committed datatype that is *not* a
/// reference is screened and passes, so a copy of an object with a named type
/// still commits beside a delete (issue #317).
///
/// Refusing every committed datatype would have been the easy way to stay safe
/// at the shared Datatype message, and would have taken this with it.
///
/// The dataset carries a committed datatype *and* an attribute whose own
/// datatype is committed, which are two different indirections: the first is a
/// shared Datatype message, the second a shared field inside an ordinary
/// Attribute message. Both have to be followed for this copy to go through.
#[test]
fn a_copy_of_a_committed_ordinary_datatype_still_commits_beside_a_delete() {
    let path = temp_path("hdf5_pure_edit_ref_copy_committed_ok.h5");
    let mut b = FileBuilder::new();
    b.commit_datatype("mytype", hdf5_pure::make_f64_type());
    b.commit_datatype("counttype", hdf5_pure::make_i32_type());
    let mut g = b.create_group("g");
    g.create_dataset("inner").with_i32_data(&[1, 2, 3]);
    b.add_group(g.finish());
    let ds = b.create_dataset("src");
    ds.with_f64_data(&[1.5, 2.5])
        .with_committed_datatype("mytype");
    ds.set_attr_committed("count", AttrValue::I32(7), "counttype");
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session.copy("src", "dup").unwrap();
        session.root().delete("g").unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("dup").unwrap().read_f64().unwrap(), [1.5, 2.5]);
    // Compared against the source's rather than against a literal: what
    // matters here is that the copy carried the attribute across, not how the
    // reader widens an `i32`.
    assert_eq!(
        file.dataset("dup").unwrap().attrs().unwrap(),
        file.dataset("src").unwrap().attrs().unwrap()
    );
    assert!(file.group("g").is_err(), "g was deleted");
    drop(file);
}

/// The chunked/extensible variable-length-string refusal is an `||` of two
/// independent conditions (`chunk_options.is_chunked()` and
/// `maxshape.is_some()`); this exercises the `with_maxshape`-alone half,
/// which the test above does not (it only sets `with_chunks`).
#[test]
fn add_extensible_vlen_string_dataset_is_rejected_without_writing() {
    let path = temp_path("hdf5_pure_edit_add_extensible_vlen_string.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        let err = session
            .root()
            .create_dataset("labels", |b| {
                b.with_vlen_strings(&["a", "b", "c"])
                    .with_maxshape(&[u64::MAX]);
            })
            .unwrap_err();
        assert!(
            err.to_string().contains("variable-length-string"),
            "got: {err}"
        );
        session.commit().unwrap();
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
}

/// A zero-element variable-length-string dataset (an empty `with_vlen_strings`
/// call) is a valid degenerate case: no global heap collection needs to be
/// placed at all, and the layout falls back to the same undefined-address
/// sentinel as any other empty dataset.
#[test]
fn add_zero_element_vlen_string_dataset_via_edit_session() {
    let path = temp_path("hdf5_pure_edit_add_zero_vlen_string.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("labels", |b| {
                b.with_vlen_strings(&[]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("labels").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![0]);
    assert_eq!(ds.read_string().unwrap(), Vec::<String>::new());
}

/// `write_dataset(...).with_vlen_strings(...)` resolves every staged element
/// reference against a global heap collection it places (issue #321).
///
/// This was a refusal until then, and the refusal existed for a reason worth
/// keeping under test: the staged references carry a **placeholder** heap
/// address of zero, and writing them as if they were final is the bug class
/// issue #105 fixed for the *add* path and issue #318 fixed for the reference
/// one. Reading the strings back is one half of the check. The other is reading
/// the element bytes and asserting no reference kept address zero — because a
/// reader that resolves a zero address to "null" would report an empty string,
/// and `read_string` alone cannot tell that from a string that is genuinely
/// empty.
#[test]
fn overwriting_a_vlen_string_dataset_leaves_no_placeholder_address() {
    let path = temp_path("hdf5_pure_edit_overwrite_vlen_no_placeholder.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("labels").with_vlen_strings(&["a", "b"]);
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("labels")
            .unwrap()
            .write_staged(|b| {
                b.with_vlen_strings(&["x", "y"]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("labels").unwrap();
    assert_eq!(
        ds.read_string().unwrap(),
        vec!["x".to_string(), "y".to_string()]
    );

    // A variable-length element reference is `[4-byte length][8-byte global
    // heap address][4-byte object index]`. Neither string is null, so every
    // address must have been patched away from the staged zero.
    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 32, "two 16-byte element references");
    let (elements, rest) = raw.as_chunks::<16>();
    assert!(
        rest.is_empty(),
        "element references are a whole number of 16"
    );
    for (i, element) in elements.iter().enumerate() {
        let addr = u64::from_le_bytes(element[4..12].try_into().unwrap());
        assert_ne!(addr, 0, "element {i} kept its placeholder heap address");
    }
}

/// The object-reference counterpart of
/// [`write_dataset_rejects_vlen_strings_without_writing`], and the regression
/// test for issue #318: both builders stage placeholder element bytes that only
/// the add path resolves, so both are refused as the overwrite is staged.
///
/// Before the refusal existed, this same sequence returned `Ok` from
/// `write_staged` **and** from `commit`, having written eight zero bytes over a
/// working reference dataset. The file-content and `dereference` assertions are
/// what make this a test about that rather than about an error message; no
/// mutation of the code as it stands reaches them, since the refusal precedes
/// every write, and they are here for a version of this code where it does not.
#[test]
fn write_dataset_rejects_object_references_without_writing() {
    let path = temp_path("hdf5_pure_edit_write_object_ref_rejected.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[1]);
    b.create_dataset("bb").with_i32_data(&[2]);
    b.create_dataset("refs").with_path_references(&["a"]);
    b.write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        let err = session
            .dataset("refs")
            .unwrap()
            .write_staged(|b| {
                b.with_path_references(&["bb"]);
            })
            .unwrap_err();
        assert!(
            err.to_string().contains("object-reference"),
            "expected an object-reference refusal, got: {err}"
        );
        assert!(
            !session.has_staged_edits(),
            "a refused overwrite must not be left staged"
        );
    }

    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "file modified on refusal"
    );
    let file = File::open(&path).unwrap();
    // Still a live reference to `a`, not the address-zero placeholder the
    // unrefused write left behind.
    let targets = file.dataset("refs").unwrap().dereference().unwrap();
    assert_eq!(targets.len(), 1);
    match &targets[0] {
        Object::Dataset(ds) => assert_eq!(ds.read_i32().unwrap(), vec![1]),
        other => panic!("expected a dataset reference, got {other:?}"),
    }
    drop(file);
}

/// The boundary of the refusal above: what it rejects is an *unresolved* target,
/// not the object-reference datatype. `with_reference_data` supplies the stored
/// addresses itself, so there is nothing for the commit to resolve and the
/// overwrite is applied.
///
/// This is the other half of issue #318's fix. Widening the refusal to "any
/// reference dataset" would take a working capability away, and the whole suite
/// would still pass.
///
/// It pins that the write *applies*, not that the addresses are sound. Nothing
/// screens them: an address into a subtree the same commit deletes is stored
/// as-is, where the same target named as a path is refused (issue #317's family,
/// which this refusal neither creates nor closes).
#[test]
fn write_dataset_accepts_resolved_reference_addresses() {
    let path = temp_path("hdf5_pure_edit_write_resolved_refs.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[1]);
    b.create_dataset("bb").with_i32_data(&[2]);
    b.create_dataset("refs").with_path_references(&["a"]);
    // A second reference dataset is how the test learns `bb`'s address without
    // reaching into the file format itself.
    b.create_dataset("refs_to_bb").with_path_references(&["bb"]);
    b.write(&path).unwrap();

    let addr_bb = {
        let file = File::open(&path).unwrap();
        let raw = file.dataset("refs_to_bb").unwrap().read_raw().unwrap();
        u64::from_le_bytes(raw[..8].try_into().unwrap())
    };

    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("refs")
            .unwrap()
            .write_staged(|b| {
                b.with_reference_data(&[addr_bb]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let targets = file.dataset("refs").unwrap().dereference().unwrap();
    assert_eq!(targets.len(), 1);
    match &targets[0] {
        Object::Dataset(ds) => assert_eq!(ds.read_i32().unwrap(), vec![2]),
        other => panic!("expected a dataset reference, got {other:?}"),
    }
    drop(file);
}

/// A staged dataset whose datatype occupies zero bytes per element is refused at
/// `commit`, alongside the other dataset guards, rather than reaching the write.
/// The whole-file writer refuses the same type; this is the second door into a
/// file, and a caller-built `Datatype` reaches it without passing through the
/// parse-side refusal (issue #268).
#[test]
fn a_zero_width_element_type_is_refused_by_a_staged_write() {
    let path = temp_path("hdf5_pure_edit_zero_width.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        let err = session
            .root()
            .create_dataset("zero_width", |b| {
                // Empty element bytes on purpose: four elements of zero width
                // *is* zero bytes, so this satisfies the shape-versus-data check
                // and reaches the chunk splitter, where the division by the
                // element size lives. Non-empty data would be refused earlier as
                // a shape mismatch and never exercise it.
                b.with_raw_data(
                    Datatype::String {
                        size: 0,
                        padding: StringPadding::NullPad,
                        charset: CharacterSet::Ascii,
                    },
                    Vec::new(),
                    4,
                )
                .with_chunks(&[2]);
            })
            .unwrap_err();
        match err {
            Error::Format(FormatError::ZeroSizedDatatype { class: 3 }) => {}
            other => panic!("expected ZeroSizedDatatype for class 3, got {other:?}"),
        }
        session.commit().unwrap();
    }

    // A refused addition leaves the file exactly as it was.
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    drop(file);
}

/// An empty (zero-element) chunked, unlimited dataset added in place: the shape
/// an incremental writer declares its schema at before any data has arrived
/// (issue #284). The edit engine used to refuse it outright, so a schema-first
/// writer had to rewrite the whole file to declare one column.
///
/// The dataset that comes out has to be the same one the whole-file writer
/// makes, which means growable: the append in the second session is what proves
/// the index over zero chunks is a real Extensible Array and not a placeholder.
#[test]
fn add_empty_extensible_chunked_dataset_and_grow_it() {
    let path = temp_path("hdf5_pure_edit_empty_chunked.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("col", |b| {
                b.with_i64_data(&[])
                    .with_shape(&[0])
                    .with_maxshape(&[u64::MAX])
                    .with_chunks(&[2]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    {
        let file = File::open(&path).unwrap();
        let ds = file.dataset("col").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![0]);
        assert_eq!(ds.read_i64().unwrap(), Vec::<i64>::new());
        // Chunked storage with the growable index, not a contiguous fallback.
        assert!(
            matches!(
                ds.layout().unwrap(),
                hdf5_pure::Layout::Chunked {
                    index: hdf5_pure::ChunkIndex::ExtensibleArray,
                    ..
                }
            ),
            "expected an extensible-array chunked layout, got {:?}",
            ds.layout().unwrap()
        );
    }

    // Two appends of two whole chunks each (the chunk is 2 elements wide), so
    // the second grows an index that already holds chunks rather than one built
    // empty. A chunk wide enough to swallow both batches would exercise the
    // first case twice and never the second.
    for batch in [[1i64, 2, 3, 4].as_slice(), [5i64, 6, 7, 8].as_slice()] {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("col")
            .unwrap()
            .append_staged(|a| {
                a.append_i64(batch);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("col").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![8]);
    assert_eq!(ds.read_i64().unwrap(), vec![1, 2, 3, 4, 5, 6, 7, 8]);
    // Four chunks of two, so the index really did grow past what it was built
    // holding.
    assert_eq!(ds.chunks().unwrap().len(), 4);
    // The file the edits were appended into is untouched.
    assert_eq!(
        file.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    drop(file);
}

/// The other empty-chunked shapes the same lifted refusal now admits: a
/// fixed-shape (non-extensible) one, a multi-dimensional one, and a filtered
/// one. Each has zero chunks, so each exercises a different index built over
/// nothing — a fixed array, and the filtered pipeline that never runs.
#[test]
fn add_empty_chunked_datasets_of_every_flavor() {
    let path = temp_path("hdf5_pure_edit_empty_chunked_flavors.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        let root = session.root();
        root.create_dataset("fixed", |b| {
            b.with_i32_data(&[]).with_shape(&[0]).with_chunks(&[8]);
        })
        .unwrap();
        root.create_dataset("two_d", |b| {
            b.with_f64_data(&[])
                .with_shape(&[0, 3])
                .with_maxshape(&[u64::MAX, 3])
                .with_chunks(&[4, 3]);
        })
        .unwrap();
        // The issue's own repro shape: a datatype and a shape, no data at all.
        // `with_i64_data(&[])` reaches the same place by a different route, and
        // only this one covers a builder whose `data` is `None`.
        root.create_dataset("declared", |b| {
            b.with_dtype(hdf5_pure::make_u64_type())
                .with_shape(&[0])
                .with_maxshape(&[u64::MAX])
                .with_chunks(&[512]);
        })
        .unwrap();
        root.create_dataset("filtered", |b| {
            b.with_i32_data(&[])
                .with_shape(&[0])
                .with_maxshape(&[u64::MAX])
                .with_chunks(&[16])
                .with_deflate(6)
                .with_shuffle();
        })
        .unwrap();
        session.commit().unwrap();
    }

    // Every flavor asserts its chunk index, not just its shape and an empty
    // read: a contiguous dataset satisfies those two just as well, so on their
    // own they cannot tell whether chunked storage was emitted at all.
    let index_of = |name: &str| match File::open(&path).unwrap().dataset(name).unwrap().layout() {
        Ok(hdf5_pure::Layout::Chunked { index, .. }) => format!("{index:?}"),
        other => panic!("[{name}] expected chunked storage, got {other:?}"),
    };
    assert_eq!(index_of("fixed"), "FixedArray");
    assert_eq!(index_of("two_d"), "ExtensibleArray");
    assert_eq!(index_of("declared"), "ExtensibleArray");
    assert_eq!(index_of("filtered"), "ExtensibleArray");

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("fixed").unwrap().shape().unwrap(), vec![0]);
    assert_eq!(
        file.dataset("fixed").unwrap().read_i32().unwrap(),
        Vec::<i32>::new()
    );
    assert_eq!(file.dataset("two_d").unwrap().shape().unwrap(), vec![0, 3]);
    assert_eq!(
        file.dataset("two_d").unwrap().read_f64().unwrap(),
        Vec::<f64>::new()
    );
    assert_eq!(file.dataset("declared").unwrap().shape().unwrap(), vec![0]);
    assert_eq!(
        file.dataset("declared").unwrap().read_u64().unwrap(),
        Vec::<u64>::new()
    );
    assert_eq!(file.dataset("filtered").unwrap().shape().unwrap(), vec![0]);
    assert_eq!(
        file.dataset("filtered").unwrap().read_i32().unwrap(),
        Vec::<i32>::new()
    );
    // The filter pipeline is recorded even though no chunk was ever encoded, so
    // the first append compresses rather than silently writing raw chunks.
    assert!(
        !file
            .dataset("filtered")
            .unwrap()
            .filter_pipeline()
            .is_empty(),
        "the filter pipeline must survive a dataset with no chunks"
    );

    // Growing the filtered one is the case an incremental writer actually runs.
    drop(file);
    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("filtered")
            .unwrap()
            .append_staged(|a| {
                a.append_i32(&(0..40).collect::<Vec<_>>());
            })
            .unwrap();
        session.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("filtered").unwrap().read_i32().unwrap(),
        (0..40).collect::<Vec<i32>>()
    );
    drop(file);
}

// ---------------------------------------------------------------------------
// Replacing an object at its own path in one commit (issue #305)
// ---------------------------------------------------------------------------

/// A file holding one dataset and one populated group, both at the root, for the
/// replacement tests below to rotate.
fn write_rotation_starter(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.create_dataset("slot").with_f64_data(&[1.0, 2.0, 3.0]);
    b.create_dataset("bystander").with_i32_data(&[42]);
    b.write(path).unwrap();

    let session = File::open_rw(path).unwrap();
    session.root().create_group("g").unwrap();
    session.root().create_group("g/sub").unwrap();
    session
        .root()
        .create_dataset("g/inner", |b| {
            b.with_i32_data(&[7, 8]);
        })
        .unwrap();
    session.commit().unwrap();
}

#[test]
fn a_dataset_is_replaced_at_its_own_path_in_one_commit() {
    let path = temp_path("hdf5_pure_edit_replace_dataset.h5");
    write_rotation_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("slot").unwrap();
        session
            .root()
            .create_dataset("slot", |b| {
                b.with_i32_data(&[9, 8, 7, 6]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    // The new object, not the old one, and not both: the datatype changed from
    // f64 to i32 and the length from 3 to 4, so a surviving original reads as
    // the wrong shape rather than as plausible data.
    let slot = file.dataset("slot").unwrap();
    assert_eq!(slot.shape().unwrap(), vec![4]);
    assert_eq!(slot.read_i32().unwrap(), vec![9, 8, 7, 6]);
    // Exactly one link named `slot` — a replacement removes the original's link
    // rather than adding a second one beside it.
    assert_eq!(
        file.root()
            .iter_datasets()
            .unwrap()
            .filter(|(n, _)| n == "slot")
            .count(),
        1
    );
    assert_eq!(
        file.dataset("bystander").unwrap().read_i32().unwrap(),
        vec![42]
    );
    assert_eq!(
        file.dataset("g/inner").unwrap().read_i32().unwrap(),
        vec![7, 8]
    );
    drop(file);
}

#[test]
fn replacing_a_group_replaces_its_whole_subtree() {
    let path = temp_path("hdf5_pure_edit_replace_group.h5");
    write_rotation_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("g").unwrap();
        session
            .root()
            .create_group_with("g", |g| {
                g.set_attr("generation", AttrValue::I64(2));
                g.create_dataset("fresh", |b| {
                    b.with_i32_data(&[1, 2, 3]);
                });
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    // The replacement is a new, empty group configured in the same commit: the
    // original's child is gone rather than inherited.
    assert!(
        file.dataset("g/inner").is_err(),
        "the replaced group kept the original's child"
    );
    assert_eq!(
        file.dataset("g/fresh").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(
        file.group("g").unwrap().attrs().unwrap()["generation"],
        AttrValue::I64(2)
    );
    assert_eq!(
        file.root()
            .iter_groups()
            .unwrap()
            .filter(|(n, _)| n == "g")
            .count(),
        1
    );
    drop(file);
}

#[test]
fn a_dataset_is_replaced_below_the_root_too() {
    let path = temp_path("hdf5_pure_edit_replace_nested.h5");
    write_rotation_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("g/inner").unwrap();
        session
            .root()
            .create_dataset("g/inner", |b| {
                b.with_i32_data(&[5, 5, 5]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("g/inner").unwrap().read_i32().unwrap(),
        vec![5, 5, 5]
    );
    drop(file);
}

#[test]
fn rotating_one_path_in_a_session_stops_growing_the_file() {
    let path = temp_path("hdf5_pure_edit_rotation_bounded.h5");
    write_rotation_starter(&path);

    let mut sizes = Vec::new();
    {
        let session = File::open_rw(&path).unwrap();
        for round in 0..12u32 {
            session.root().delete("slot").unwrap();
            session
                .root()
                .create_dataset("slot", |b| {
                    b.with_i32_data(&vec![round as i32; 256]);
                })
                .unwrap();
            session.commit().unwrap();
            sizes.push(session.file_size());
        }
    }

    // A rotation is what the refusal used to cost two commits, and the point of
    // paying it once is that the file reaches a steady state instead of growing
    // by a dataset per round: each commit frees the object the previous one
    // placed, and the next reuses that space. Judged against the *second* round
    // rather than the first, because the first has nothing freed to draw on yet.
    let ceiling = sizes[1];
    assert!(
        sizes[2..].iter().all(|&s| s <= ceiling),
        "the file grew past its second-round size under rotation: {sizes:?}"
    );

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("slot").unwrap().read_i32().unwrap(),
        vec![11i32; 256]
    );
    drop(file);
}

#[test]
fn an_addition_below_a_path_needs_that_path_replaced_too() {
    // Replacing `g` makes an addition below it unambiguous: `g/added` is placed
    // in the group this commit builds, not in the one it removes.
    let path = temp_path("hdf5_pure_edit_replace_descendant_ok.h5");
    write_rotation_starter(&path);
    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("g").unwrap();
        session.root().create_group("g").unwrap();
        session
            .root()
            .create_dataset("g/added", |b| {
                b.with_i32_data(&[1, 2]);
            })
            .unwrap();
        session.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("g/added").unwrap().read_i32().unwrap(),
        vec![1, 2]
    );
    assert!(
        file.dataset("g/inner").is_err(),
        "the addition landed in the group being removed rather than its replacement"
    );
    drop(file);

    // Without that replacement it is the ambiguity the refusal exists for: the
    // addition names a group whose own link this commit removes.
    let path = temp_path("hdf5_pure_edit_replace_descendant.h5");
    write_rotation_starter(&path);
    let before = std::fs::read(&path).unwrap();
    let session = File::open_rw(&path).unwrap();
    session.root().delete("g").unwrap();
    session
        .root()
        .create_dataset("g/added", |b| {
            b.with_i32_data(&[1]);
        })
        .unwrap();
    let err = session.commit().unwrap_err();
    assert!(
        matches!(&err, Error::EditUnsupported(m) if m.contains("overlaps an addition")),
        "unexpected error: {err:?}"
    );
    // The refusal is a preflight, so not a byte of the file changed.
    drop(session);
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn a_group_and_a_dataset_replace_each_other() {
    let path = temp_path("hdf5_pure_edit_replace_kind_swap.h5");
    write_rotation_starter(&path);

    // A dataset over a group: the whole subtree goes with the link.
    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("g").unwrap();
        session
            .root()
            .create_dataset("g", |b| {
                b.with_i32_data(&[4, 5, 6]);
            })
            .unwrap();
        session.commit().unwrap();
    }
    {
        let file = File::open(&path).unwrap();
        assert_eq!(
            file.dataset("g").unwrap().read_i32().unwrap(),
            vec![4, 5, 6]
        );
        // `g` names a dataset now, so opening it as a group is refused at the
        // lookup (issue #352) rather than handed back as a handle whose calls
        // fail one at a time.
        assert!(
            matches!(file.group("g"), Err(Error::NotAGroup(_))),
            "`g` still opens as a group"
        );
        assert!(file.dataset("g/inner").is_err());
        assert!(file.group("g/sub").is_err());
    }

    // And a group back over that dataset.
    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("g").unwrap();
        session
            .root()
            .create_group_with("g", |g| {
                g.create_dataset("back", |b| {
                    b.with_i32_data(&[7]);
                });
            })
            .unwrap();
        session.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    // The mirror of the refusal above, and the same shape: a kind, not an
    // absence.
    assert!(
        matches!(file.dataset("g"), Err(Error::NotADataset(_))),
        "`g` still resolves as a dataset"
    );
    assert_eq!(file.dataset("g/back").unwrap().read_i32().unwrap(), vec![7]);
    drop(file);
}

#[test]
fn a_staged_edit_to_a_replaced_object_is_refused() {
    let path = temp_path("hdf5_pure_edit_replace_edited.h5");
    write_rotation_starter(&path);

    // `g` is replaced by a *dataset*, so the group-attribute edit at the same
    // path can only mean the group being removed. A replacement makes every
    // path under it name the new object, and there is no new group here to
    // carry the attribute.
    let session = File::open_rw(&path).unwrap();
    // Taken before the replacement is staged: once it is, `g` names the staged
    // dataset, so this is the only way to hold the group being removed.
    let doomed = session.group("g").unwrap();
    session.root().delete("g").unwrap();
    session
        .root()
        .create_dataset("g", |b| {
            b.with_i32_data(&[1]);
        })
        .unwrap();
    assert!(
        matches!(session.group("g"), Err(Error::NotAGroup(_))),
        "`g` names the staged dataset before the commit, not the group it replaces"
    );
    doomed.set_attr("generation", AttrValue::I64(1)).unwrap();
    let err = session.commit().unwrap_err();
    assert!(
        matches!(&err, Error::EditUnsupported(m) if m.contains("at or under a replaced path")),
        "unexpected error: {err:?}"
    );
    // A handle keeps the session (and its exclusive lock) alive until it is
    // dropped; on Windows that lock is mandatory, so the reopen below needs
    // both gone.
    drop(doomed);
    drop(session);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("g/inner").unwrap().read_i32().unwrap(),
        vec![7, 8]
    );
    drop(file);

    // The same rule one level down: `g` is replaced by a group this time, but
    // `g/sub` names the *original's* subgroup, which the replacement does not
    // create. Refused for the same reason and by the same guard.
    let path = temp_path("hdf5_pure_edit_replace_edited_child.h5");
    write_rotation_starter(&path);
    let before = std::fs::read(&path).unwrap();
    let session = File::open_rw(&path).unwrap();
    session.root().delete("g").unwrap();
    session.root().create_group("g").unwrap();
    session
        .group("g/sub")
        .unwrap()
        .set_attr("generation", AttrValue::I64(1))
        .unwrap();
    let err = session.commit().unwrap_err();
    assert!(
        matches!(&err, Error::EditUnsupported(m) if m.contains("at or under a replaced path")),
        "unexpected error: {err:?}"
    );
    drop(session);
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn a_copy_replaces_an_object_at_its_own_path() {
    let path = temp_path("hdf5_pure_edit_replace_by_copy.h5");
    write_rotation_starter(&path);

    // A copy is linked into its parent by the same apply-loop step as a created
    // dataset, and after the same removal, so it replaces a path too.
    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("slot").unwrap();
        session.copy("g/inner", "slot").unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("slot").unwrap().read_i32().unwrap(),
        vec![7, 8]
    );
    // The copy's source is untouched, and the replaced original is gone rather
    // than shadowed: it held f64, so a surviving link reads as the wrong type.
    assert_eq!(
        file.dataset("g/inner").unwrap().read_i32().unwrap(),
        vec![7, 8]
    );
    assert_eq!(file.dataset("slot").unwrap().shape().unwrap(), vec![2]);
    drop(file);
}

#[test]
fn a_copy_reading_from_a_replaced_path_is_refused() {
    let path = temp_path("hdf5_pure_edit_replace_copy_source.h5");
    write_rotation_starter(&path);
    let before = std::fs::read(&path).unwrap();

    // A copy takes its bytes from the pre-commit file. Reading from a path this
    // same commit replaces would put the *original* at `backup` while the
    // replacement lands at `slot`, so one commit would produce two different
    // objects from one path with nothing to say so.
    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("slot").unwrap();
        session
            .root()
            .create_dataset("slot", |b| {
                b.with_i32_data(&[9, 9, 9, 9]);
            })
            .unwrap();
        session.copy("slot", "backup").unwrap();
        let err = session.commit().unwrap_err();
        assert!(
            matches!(&err, Error::EditUnsupported(m) if m.contains("reads from a path the same commit replaces")),
            "unexpected error: {err:?}"
        );
        drop(session);
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    // Reading from inside a replaced *group* is the same conflict one level down.
    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("g").unwrap();
        session
            .root()
            .create_group_with("g", |g| {
                g.create_dataset("inner", |b| {
                    b.with_i32_data(&[7, 7, 7, 7, 7]);
                });
            })
            .unwrap();
        session.copy("g/inner", "backup").unwrap();
        let err = session.commit().unwrap_err();
        assert!(
            matches!(&err, Error::EditUnsupported(m) if m.contains("reads from a path the same commit replaces")),
            "unexpected error: {err:?}"
        );
        drop(session);
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    // A source that is deleted but *not* replaced is unambiguous — that is a
    // move — and stays allowed.
    {
        let session = File::open_rw(&path).unwrap();
        session.copy("slot", "moved").unwrap();
        session.root().delete("slot").unwrap();
        session.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("moved").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert!(file.dataset("slot").is_err());
    drop(file);
}

/// A commit refused partway through its preflight must leave the staged set
/// exactly as it found it (issue #316).
///
/// Three of the four edits staged here are ones the commit never objects to;
/// the deletion/addition pair is the one it refuses. The refusal fires in the
/// delete-staging loop, which runs after the other pending vectors have been
/// drained, so before the fix `commit()` returned `Err` having destroyed three
/// valid edits and left `has_staged_edits()` answering `false`.
#[test]
fn a_refused_commit_leaves_its_other_staged_edits_alone() {
    let path = temp_path("hdf5_pure_refused_keeps_staged.h5");
    {
        let mut b = FileBuilder::new();
        let mut g = b.create_group("g");
        g.create_dataset("inner").with_i32_data(&[1, 2, 3]);
        b.add_group(g.finish());
        b.write(&path).unwrap();
    }
    let before = std::fs::read(&path).unwrap();

    let session = File::open_rw(&path).unwrap();
    session
        .root()
        .create_dataset("added_a", |b| {
            b.with_i32_data(&[10]);
        })
        .unwrap();
    session.root().create_group("newgrp").unwrap();
    session
        .root()
        .set_attr("label", AttrValue::String("v1".into()))
        .unwrap();
    // The refused pair: a deletion that overlaps an addition under it.
    session.root().delete("g").unwrap();
    session
        .root()
        .create_dataset("g/x", |b| {
            b.with_i32_data(&[1]);
        })
        .unwrap();

    let err = session.commit().unwrap_err();
    assert!(
        matches!(&err, Error::EditUnsupported(m) if m.contains("deletion overlaps an addition")),
        "unexpected error: {err:?}"
    );
    assert!(
        session.has_staged_edits(),
        "the refusal discarded the staged set"
    );

    // The batch is still whole, so committing it again refuses identically
    // rather than applying the three edits the refusal was not about.
    let again = session.commit().unwrap_err();
    assert_eq!(format!("{err:?}"), format!("{again:?}"));

    drop(session);
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "a refused commit wrote to the file"
    );
}

/// A commit refused in its *first* preflight loop must not leave the rest of
/// the batch staged for a later `commit()` to apply on its own (issue #316).
///
/// The overwrite below is refused by the write preflight, which runs before
/// anything else. Before the fix only `pending_writes` had been drained, so the
/// caller was told the commit failed atomically and a later `commit()` — for
/// unrelated work, perhaps much later — silently applied the surviving half of
/// the batch it had been told was rejected.
#[test]
fn a_refused_commit_does_not_apply_its_survivors_later() {
    let path = temp_path("hdf5_pure_refused_no_partial_apply.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
        b.write(&path).unwrap();
    }
    let before = std::fs::read(&path).unwrap();

    let session = File::open_rw(&path).unwrap();
    session
        .root()
        .create_dataset("added_a", |b| {
            b.with_i32_data(&[10]);
        })
        .unwrap();
    // Refused at commit: a value overwrite is not a reshape.
    session
        .dataset("keep")
        .unwrap()
        .write_staged(|b| {
            b.with_i32_data(&[1, 2, 3, 4, 5]);
        })
        .unwrap();

    let err = session.commit().unwrap_err();
    assert!(
        matches!(&err, Error::EditUnsupported(m) if m.contains("shape does not match")),
        "unexpected error: {err:?}"
    );
    let again = session.commit().unwrap_err();
    assert_eq!(
        format!("{err:?}"),
        format!("{again:?}"),
        "the second commit applied the refused batch's survivors"
    );

    drop(session);
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "a refused commit wrote to the file"
    );
    let file = File::open(&path).unwrap();
    assert!(
        file.dataset("added_a").is_err(),
        "an edit from a refused batch reached the file"
    );
}

/// The restore a refused commit performs covers *every* kind of staged edit,
/// not the ones that happen to be drained late (issue #316).
///
/// Each case below stages exactly one edit of one kind, plus the deletion it
/// conflicts with, so the commit's refusal depends on that edit still being
/// there. A second `commit()` therefore discriminates: with the staged set put
/// back whole it refuses identically, and with that one kind dropped it would
/// see a lone deletion and *succeed*, taking `g` with it.
#[test]
fn a_refused_commit_restores_every_kind_of_staged_edit() {
    /// Stages the one edit this case is about, against a session whose batch
    /// already holds `delete("g")`.
    type Stage = fn(&File, &File);

    let cases: &[(&str, Stage)] = &[
        ("dataset addition", |s, _| {
            s.root()
                .create_dataset("g/added", |b| {
                    b.with_i32_data(&[1]);
                })
                .unwrap();
        }),
        ("group creation", |s, _| {
            s.root().create_group("g/sub").unwrap();
        }),
        ("group attribute", |s, _| {
            s.group("g")
                .unwrap()
                .set_attr("tag", AttrValue::I32(1))
                .unwrap();
        }),
        ("in-file copy", |s, _| {
            s.copy("original", "g/copied").unwrap();
        }),
        ("cross-file copy", |s, other| {
            s.copy_from(other, "donor", "g/from_other").unwrap();
        }),
        ("value overwrite", |s, _| {
            s.dataset("g/inner")
                .unwrap()
                .write_staged(|b| {
                    b.with_i32_data(&[9, 9, 9, 9]);
                })
                .unwrap();
        }),
        ("dataset attribute", |s, _| {
            s.dataset("g/inner")
                .unwrap()
                .set_attr("tag", AttrValue::I32(1))
                .unwrap();
        }),
        ("append", |s, _| {
            s.dataset("g/inner")
                .unwrap()
                .append_staged(|b| {
                    b.append_i32(&[5, 6]);
                })
                .unwrap();
        }),
        ("second deletion", |s, _| {
            s.root().delete("g/inner").unwrap();
        }),
    ];

    let donor_path = temp_path("hdf5_pure_restore_kinds_donor.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("donor").with_i32_data(&[7]);
        b.write(&donor_path).unwrap();
    }

    for (what, stage) in cases {
        let path = temp_path(&format!("hdf5_pure_restore_{}.h5", what.replace(' ', "_")));
        {
            let mut b = FileBuilder::new();
            b.create_dataset("original").with_i32_data(&[0]);
            let mut g = b.create_group("g");
            g.create_dataset("inner")
                .with_i32_data(&[1, 2, 3, 4])
                .with_chunks(&[2])
                .with_maxshape(&[u64::MAX]);
            b.add_group(g.finish());
            b.write(&path).unwrap();
        }
        let before = std::fs::read(&path).unwrap();

        {
            let donor = File::open(&donor_path).unwrap();
            let session = File::open_rw(&path).unwrap();
            session.root().delete("g").unwrap();
            stage(&session, &donor);

            let Err(first) = session.commit() else {
                panic!("[{what}] the commit was expected to refuse");
            };
            assert!(
                matches!(
                    &first,
                    Error::EditUnsupported(_) | Error::AppendUnsupported(_)
                ),
                "[{what}] unexpected error: {first:?}"
            );
            let Err(second) = session.commit() else {
                panic!("[{what}] the second commit applied the batch the first refused");
            };
            assert_eq!(
                format!("{first:?}"),
                format!("{second:?}"),
                "[{what}] the refusal did not put this edit back"
            );
        }

        assert_eq!(
            std::fs::read(&path).unwrap(),
            before,
            "[{what}] a refused commit wrote to the file"
        );
    }
}

/// A staging call that refuses stages nothing, even when it carries a batch.
///
/// `create_group_with` records a group, its attributes and its whole subtree in
/// one call, and each dataset is validated as it is staged — so the refusal
/// below arrives with three edits already recorded. They are dropped with it
/// (issue #316), leaving the session as the call found it.
#[test]
fn a_refused_staging_call_stages_none_of_its_batch() {
    let path = temp_path("hdf5_pure_refused_staging_batch.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        // One edit staged beforehand: the rewind must not reach past the call
        // that failed.
        session
            .root()
            .create_dataset("kept", |b| {
                b.with_i32_data(&[1]);
            })
            .unwrap();

        let err = session
            .root()
            .create_group_with("batch", |g| {
                g.set_attr("tag", AttrValue::I32(1));
                g.create_dataset("good", |b| {
                    b.with_i32_data(&[1, 2]);
                });
                // Data that does not match the shape: refused as it is staged.
                g.create_dataset("bad", |b| {
                    b.with_f64_data(&[9.0, 9.0, 9.0]).with_shape(&[7]);
                });
            })
            .unwrap_err();
        assert!(
            matches!(&err, Error::EditUnsupported(m) if m.contains("shape")),
            "unexpected error: {err:?}"
        );

        session.commit().unwrap();
    }

    // Only the edit staged before the refused call reached the file.
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("kept").unwrap().read_i32().unwrap(), vec![1]);
    assert!(file.group("batch").is_err(), "the refused group was staged");
    assert!(
        file.dataset("batch/good").is_err(),
        "a dataset from the refused call was staged"
    );
    drop(file);
    assert_ne!(std::fs::read(&path).unwrap(), before);
}

/// A group's added datasets are placed in staging order, which is the order
/// `preflight_reference_targets` replays.
///
/// Two *reference* datasets in one group sort together (`sort_by_key` is
/// stable), so staging order alone decides which is placed first and therefore
/// whether the second can resolve the first. That makes the agreement between
/// the preflight's replay and the apply loop's own placement observable: if
/// they disagreed, this commit would pass every guard and then fail with
/// "still writing" having already written part of itself.
#[test]
fn a_reference_datasets_placement_order_is_the_order_the_preflight_proved() {
    let path = temp_path("hdf5_pure_edit_ref_placement_order.h5");
    write_starter(&path);

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("first", |b| {
                b.with_path_references(&["original"]);
            })
            .unwrap();
        session
            .root()
            .create_dataset("second", |b| {
                b.with_path_references(&["first"]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let via_second = file.dataset("second").unwrap().dereference().unwrap();
    assert_eq!(via_second.len(), 1);
    match &via_second[0] {
        Object::Dataset(ds) => {
            let via_first = ds.dereference().unwrap();
            assert_eq!(via_first.len(), 1);
            match &via_first[0] {
                Object::Dataset(d) => {
                    assert_eq!(d.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
                }
                other => panic!("expected a dataset reference, got {other:?}"),
            }
        }
        other => panic!("expected a dataset reference, got {other:?}"),
    }
}

/// A group gains its new links in the order the commit places them: copied
/// objects first — in-file copies, then cross-file ones — and then the
/// datasets added to it.
///
/// The names are chosen so that order is not the alphabetical one, and not the
/// reverse of it either, so the assertion cannot pass by coincidence. It is
/// pinned because the commit's cross-file copies join their group's other
/// copies at the point of no return rather than while the plan is built, and
/// nothing else observes where in the sequence they land.
#[test]
fn a_group_gains_its_new_links_in_placement_order() {
    let donor_path = temp_path("hdf5_pure_link_order_donor.h5");
    let path = temp_path("hdf5_pure_link_order.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("donated").with_i32_data(&[3]);
        b.write(&donor_path).unwrap();
    }
    {
        let mut b = FileBuilder::new();
        b.create_dataset("source").with_i32_data(&[1]);
        b.write(&path).unwrap();
    }

    {
        let donor = File::open(&donor_path).unwrap();
        let session = File::open_rw(&path).unwrap();
        session.root().create_group("g").unwrap();
        session
            .root()
            .create_dataset("g/z_added", |b| {
                b.with_i32_data(&[2]);
            })
            .unwrap();
        session.copy("source", "g/m_copied").unwrap();
        session
            .copy_from(&donor, "donated", "g/a_from_other")
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.group("g").unwrap().datasets().unwrap(),
        vec![
            "m_copied".to_string(),
            "a_from_other".to_string(),
            "z_added".to_string()
        ]
    );
    drop(file);
}

/// A zero-element contiguous dataset — one this crate writes itself, with no
/// help from the reference library — copies rather than being refused as out of
/// bounds (issue #336).
///
/// The issue reached this through a dataset the C library created and never
/// wrote, but the encoding is the same one `FileBuilder` produces for a shape
/// with no elements: there is nothing to store, so the layout message carries the
/// undefined address. That makes this the half of the defect reachable with no
/// reference library in the picture at all.
#[test]
fn a_zero_element_dataset_copies_as_the_storage_it_never_had() {
    let path = temp_path("hdf5_pure_edit_copy_zero_element.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("empty")
            .with_i32_data(&[])
            .with_shape(&[0]);
        b.create_dataset("full").with_i32_data(&[1, 2, 3]);
        b.write(&path).unwrap();
    }
    // The fixture's premise: no data block, which is what the copy has to carry.
    {
        let f = File::open(&path).unwrap();
        assert!(
            matches!(
                f.dataset("empty").unwrap().layout().unwrap(),
                hdf5_pure::Layout::Contiguous { address: None, .. }
            ),
            "the fixture is meant to store nothing"
        );
    }

    {
        let session = File::open_rw(&path).unwrap();
        session.copy("empty", "empty_copy").unwrap();
        // Beside a dataset that does hold data, so the same commit proves the
        // allocated path still places and repoints its block.
        session.copy("full", "full_copy").unwrap();
        session.commit().unwrap();
    }

    let f = File::open(&path).unwrap();
    let copied = f.dataset("empty_copy").unwrap();
    assert!(
        matches!(
            copied.layout().unwrap(),
            hdf5_pure::Layout::Contiguous { address: None, .. }
        ),
        "the copy materialized storage the source never had: {:?}",
        copied.layout().unwrap()
    );
    assert_eq!(copied.shape().unwrap(), vec![0], "shape carried across");
    assert_eq!(copied.read_i32().unwrap(), Vec::<i32>::new());
    assert_eq!(
        f.dataset("full_copy").unwrap().read_i32().unwrap(),
        vec![1, 2, 3],
        "the dataset that does store data copied it"
    );
}

/// Issue #321: `Dataset::write_staged` with `with_vlen_strings` overwrites a
/// variable-length-string dataset's values, resolving the staged placeholder
/// element references against a freshly placed global heap collection.
///
/// The replacement strings are deliberately *longer* than the originals: the
/// element bytes are fixed-width references either way, so the data block keeps
/// its length and only the heap collection grows. That is what makes this a
/// `WritePlan::InPlace` — not to be confused with the *commit's* in-place fast
/// path, which a staged variable-length overwrite is deliberately excluded
/// from.
#[test]
fn overwriting_a_vlen_string_dataset_replaces_its_strings() {
    let path = temp_path("hdf5_pure_edit_overwrite_vlen_contiguous.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("labels")
        .with_vlen_strings(&["a", "b", "c"]);
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("labels")
            .unwrap()
            .write_staged(|b| {
                b.with_vlen_strings(&["alpha", "beta", "gamma"]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("labels").unwrap().read_string().unwrap(),
        vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()]
    );
}

/// The chunked counterpart of
/// [`overwriting_a_vlen_string_dataset_replaces_its_strings`]. A chunked
/// variable-length-string dataset is what this crate writes for one created
/// with chunks (issue #109), so refusing to overwrite it would leave a hole in a
/// shape the crate itself produces.
#[test]
fn overwriting_a_chunked_vlen_string_dataset_replaces_its_strings() {
    let path = temp_path("hdf5_pure_edit_overwrite_vlen_chunked.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("labels")
        .with_vlen_strings(&["a", "b", "c", "d"])
        .with_chunks(&[2]);
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("labels")
            .unwrap()
            .write_staged(|b| {
                b.with_vlen_strings(&["alpha", "beta", "gamma", "delta"]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("labels").unwrap().read_string().unwrap(),
        vec![
            "alpha".to_string(),
            "beta".to_string(),
            "gamma".to_string(),
            "delta".to_string()
        ]
    );
}

/// A commit that refuses must not have placed the global heap collection a
/// staged variable-length overwrite stages (issue #321).
///
/// Resolving those references allocates, and `commit`'s preflight only reads —
/// that is what lets a refused commit restore the staged set whole (issue #316)
/// and cost the session nothing. So the plan carries the bytes unresolved and
/// the resolving happens in the apply phase. Staging a *second* overwrite that
/// only the preflight can refuse — a shape the on-disk dataset does not have —
/// is what puts a valid variable-length overwrite in front of a refusal.
///
/// The file bytes are the assertion. Placing the collection during the preflight
/// would append it and grow the file, leaving the strings of an overwrite that
/// never happened behind on every attempt.
#[test]
fn a_refused_commit_places_no_collection_for_a_staged_vlen_overwrite() {
    let path = temp_path("hdf5_pure_edit_refused_vlen_overwrite.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("labels").with_vlen_strings(&["a", "b"]);
    b.create_dataset("nums").with_i32_data(&[1, 2, 3]);
    b.write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("labels")
            .unwrap()
            .write_staged(|b| {
                b.with_vlen_strings(&["much-longer-one", "much-longer-two"]);
            })
            .unwrap();
        // Refused by the preflight, which reads the on-disk shape to find out.
        session
            .dataset("nums")
            .unwrap()
            .write_staged(|b| {
                b.with_i32_data(&[1, 2, 3, 4, 5]);
            })
            .unwrap();

        let err = session.commit().unwrap_err();
        assert!(err.to_string().contains("shape"), "got: {err}");
        assert!(
            session.has_staged_edits(),
            "a refused commit must give the staged set back (issue #316)"
        );
    }

    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "a refused commit must not have placed the overwrite's heap collection"
    );
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("labels").unwrap().read_string().unwrap(),
        vec!["a".to_string(), "b".to_string()]
    );
}

/// A chunked variable-length overwrite relocates its chunk storage
/// (`MovingWrite::ChunkedStaged`), so the blocks it vacates must be reclaimed —
/// exactly as the non-staged `MovingWrite::Chunked` relocation reclaims its own.
///
/// Asserted through the free list rather than through the file size, because
/// the file grows either way: the *global heap collections* the old strings
/// lived in are deliberately never reclaimed (a collection can be shared
/// between objects), so a size trend cannot separate the chunk storage from
/// them. `repack` is what recovers the heap.
#[test]
fn overwriting_a_chunked_vlen_string_dataset_reclaims_its_old_chunk_storage() {
    let path = temp_path("hdf5_pure_edit_overwrite_vlen_reclaim.h5");
    // 512 elements of 16-byte references is 8 KiB of chunk data, well clear of
    // the object-header slack a commit also frees.
    let before: Vec<String> = (0..512).map(|i| format!("before-{i}")).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("labels")
        .with_vlen_strings(&before.iter().map(String::as_str).collect::<Vec<_>>())
        .with_chunks(&[64]);
    b.write(&path).unwrap();

    let after: Vec<String> = (0..512).map(|i| format!("after-{i}")).collect();
    let session = File::open_rw(&path).unwrap();
    session
        .dataset("labels")
        .unwrap()
        .write_staged(|b| {
            b.with_vlen_strings(&after.iter().map(String::as_str).collect::<Vec<_>>());
        })
        .unwrap();
    session.commit().unwrap();

    let free = session.space_accounting().unwrap().reusable_free_bytes;
    assert!(
        free >= 512 * 16,
        "the vacated chunk storage was not reclaimed: {free} free bytes"
    );

    // Windows holds the file lock until the session is dropped.
    drop(session);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("labels").unwrap().read_string().unwrap(),
        after
    );
}

/// Rotating a variable-length dataset's strings in one session reaches a steady
/// state instead of growing the file by the whole payload every commit
/// (issue #321).
///
/// Each overwrite places a fresh global heap collection for the new strings.
/// Nothing else in the editor reclaims a collection, and for good reason — one
/// can be shared between objects — so before this the old strings were simply
/// left behind and only `repack` recovered them. What makes these reclaimable
/// is provenance: this session placed them, for this dataset, and
/// `invalidate_heap_provenance` drops the record the moment anything could name
/// them twice.
///
/// Judged from the second round, like the fixed-size rotation test above: the
/// first has nothing freed to draw on yet.
#[test]
fn rotating_a_vlen_dataset_in_a_session_stops_growing_the_file() {
    let path = temp_path("hdf5_pure_edit_vlen_rotation_bounded.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("labels")
        .with_vlen_strings(&["seed-a", "seed-b", "seed-c", "seed-d"]);
    b.write(&path).unwrap();

    let mut sizes = Vec::new();
    {
        let session = File::open_rw(&path).unwrap();
        for round in 0..12u32 {
            // Long enough that a leaked generation is unmistakable against the
            // object-header slack a commit also churns.
            let data: Vec<String> = (0..4)
                .map(|i| format!("round-{round}-element-{i}-{}", "x".repeat(64)))
                .collect();
            session
                .dataset("labels")
                .unwrap()
                .write_staged(|b| {
                    b.with_vlen_strings(&data.iter().map(String::as_str).collect::<Vec<_>>());
                })
                .unwrap();
            session.commit().unwrap();
            sizes.push(session.file_size());
        }
    }

    let ceiling = sizes[1];
    assert!(
        sizes[2..].iter().all(|&s| s <= ceiling),
        "the file grew past its second-round size under rotation: {sizes:?}"
    );

    let file = File::open(&path).unwrap();
    let last = file.dataset("labels").unwrap().read_string().unwrap();
    assert!(
        last[0].starts_with("round-11-element-0"),
        "got: {}",
        last[0]
    );
}

/// The reclaim must not fire when the session has done something that could name
/// a recorded collection a second time (issue #321).
///
/// An in-file `copy` of a variable-length dataset re-emits its element
/// references **verbatim**, so the copy names the very collections the source
/// does — with every heap object's reference count still 1, which is why the
/// format cannot be asked. Freeing them on the next overwrite of the source
/// would hand out space the copy still reads, and every checksum in the file
/// would still verify.
///
/// This is the test that a provenance record outliving its proof corrupts data,
/// so it asserts the *copy's* contents, not the source's.
///
/// Freeing alone would not show it: a freed region keeps its bytes until
/// something draws on it, so reading the copy straight after the bad free still
/// answers correctly. The rounds after the copy are what make the failure
/// visible — every string is the same length, so a reclaimed collection is
/// exactly the right size for the next one to be placed in, and the copy's
/// strings are overwritten by a later generation's.
#[test]
fn a_copy_stops_a_vlen_overwrite_from_reclaiming_the_shared_collection() {
    let path = temp_path("hdf5_pure_edit_vlen_copy_blocks_reclaim.h5");
    let generation = |n: u32| -> Vec<String> {
        (0..2)
            .map(|i| format!("generation-{n:02}-element-{i}"))
            .collect()
    };

    let mut b = FileBuilder::new();
    b.create_dataset("labels")
        .with_vlen_strings(&generation(0).iter().map(String::as_str).collect::<Vec<_>>());
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        let overwrite = |n: u32| {
            let data = generation(n);
            session
                .dataset("labels")
                .unwrap()
                .write_staged(|b| {
                    b.with_vlen_strings(&data.iter().map(String::as_str).collect::<Vec<_>>());
                })
                .unwrap();
            session.commit().unwrap();
        };

        // Round one records the collections it places for "labels".
        overwrite(1);

        // The copy now names those same collections.
        session.copy("labels", "clone").unwrap();
        session.commit().unwrap();

        // Every later round places a collection of exactly the size the
        // generation-1 one occupies. If the copy did not drop the record, round
        // two frees it and one of these is placed on top of the copy's strings.
        for n in 2..8 {
            overwrite(n);
        }
    }

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("clone").unwrap().read_string().unwrap(),
        generation(1),
        "the copy's strings were freed out from under it and then written over"
    );
    assert_eq!(
        file.dataset("labels").unwrap().read_string().unwrap(),
        generation(7)
    );
}

/// The other way a second name for a recorded collection is made: a **raw-bytes
/// write** (issue #321).
///
/// `with_raw_data` hands the engine element bytes it does not interpret, so a
/// caller that reads one variable-length dataset's references and stages them as
/// another dataset's data has aliased the collections they name — with every
/// heap object's reference count still 1, exactly as an in-file `copy` does.
/// This is the half of `invalidate_heap_provenance` that
/// [`a_copy_stops_a_vlen_overwrite_from_reclaiming_the_shared_collection`] does
/// not reach: the screen on the staged datatype rather than on the copy lists.
///
/// Built like that test, and for the same reason: the *aliasing* dataset's
/// contents are the assertion, and the rounds after it are what make a bad free
/// visible — every generation is the same length, so a wrongly reclaimed
/// collection is exactly the right size for the next one to be placed in.
#[test]
fn a_raw_bytes_write_stops_a_vlen_overwrite_from_reclaiming_its_collection() {
    let path = temp_path("hdf5_pure_edit_vlen_raw_blocks_reclaim.h5");
    let generation = |n: u32| -> Vec<String> {
        (0..2)
            .map(|i| format!("generation-{n:02}-element-{i}"))
            .collect()
    };

    let mut b = FileBuilder::new();
    b.create_dataset("labels")
        .with_vlen_strings(&generation(0).iter().map(String::as_str).collect::<Vec<_>>());
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        let overwrite = |n: u32| {
            let data = generation(n);
            session
                .dataset("labels")
                .unwrap()
                .write_staged(|b| {
                    b.with_vlen_strings(&data.iter().map(String::as_str).collect::<Vec<_>>());
                })
                .unwrap();
            session.commit().unwrap();
        };

        // Round one records the collections it places for "labels".
        overwrite(1);

        // Read the element references out and stage them as a second dataset's
        // data. Read before staging: re-entering the session inside the builder
        // closure would deadlock (issue #200).
        let (dt, raw) = {
            let ds = session.dataset("labels").unwrap();
            (ds.datatype().unwrap(), ds.read_raw().unwrap())
        };
        session
            .root()
            .create_dataset("alias", |b| {
                b.with_raw_data(dt.clone(), raw.clone(), 2);
            })
            .unwrap();
        session.commit().unwrap();

        for n in 2..8 {
            overwrite(n);
        }
    }

    let file = File::open(&path).unwrap();
    // Read fallibly: space handed back and reused stops being a heap collection
    // at all, so the wrong answer here is as often an error as it is a string.
    let alias = file.dataset("alias").unwrap().read_string();
    assert!(
        alias.as_ref().is_ok_and(|got| *got == generation(1)),
        "the aliased strings were freed out from under the raw-bytes dataset \
         and written over: {alias:?}"
    );
    assert_eq!(
        file.dataset("labels").unwrap().read_string().unwrap(),
        generation(7)
    );
}

/// A builder's variable-length staging describes the bytes in its `data` field,
/// so replacing those bytes must drop it (issue #321).
///
/// This is the door that made the screen in
/// [`a_raw_bytes_write_stops_a_vlen_overwrite_from_reclaiming_its_collection`]
/// skippable. `with_raw_data` used to leave `vl_string_staging` in place, so a
/// builder could carry a staging that owned one patch offset while `data` held a
/// whole new array: the commit patched element 0 against a fresh collection and
/// wrote the caller's own bytes into the rest. Because the builder still had a
/// staging, the screen's `vl_string_staging.is_none()` test skipped the entry
/// and the reclaim went ahead over an alias it had not seen.
///
/// Asserted on the *aliased* dataset after later rounds have had the chance to
/// reuse the freed space, for the same reason as the tests above: a free alone
/// leaves the bytes where they are.
#[test]
fn replacing_a_builders_data_drops_the_vlen_staging_that_described_it() {
    let path = temp_path("hdf5_pure_edit_raw_data_drops_staging.h5");
    let generation =
        |n: u32| -> Vec<String> { (0..2).map(|i| format!("gen-{n:02}-element-{i}")).collect() };

    let mut b = FileBuilder::new();
    b.create_dataset("labels")
        .with_vlen_strings(&generation(1).iter().map(String::as_str).collect::<Vec<_>>());
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        let (dt, raw) = {
            let ds = session.dataset("labels").unwrap();
            (ds.datatype().unwrap(), ds.read_raw().unwrap())
        };

        // A builder carrying *both* a staging and raw bytes lifted from
        // "labels". Its element 1 is "labels"'s own element reference.
        session
            .root()
            .create_dataset("alias", |b| {
                b.with_shape(&[2]);
                b.with_vlen_strings(&["only-one"]);
                b.with_raw_data(dt, raw, 2);
            })
            .unwrap();
        session.commit().unwrap();

        for n in 2..10 {
            let data = generation(n);
            session
                .dataset("labels")
                .unwrap()
                .write_staged(|b| {
                    b.with_vlen_strings(&data.iter().map(String::as_str).collect::<Vec<_>>());
                })
                .unwrap();
            session.commit().unwrap();
        }
    }

    let file = File::open(&path).unwrap();
    // Fallible for the same reason as the sibling test: reused space stops
    // being a heap collection at all, so the wrong answer is as often an error
    // as a string.
    let alias = file.dataset("alias").unwrap().read_string();
    assert!(
        alias.as_ref().is_ok_and(|got| got[1] == generation(1)[1]),
        "the aliased element was freed out from under it and written over: {alias:?}"
    );
    assert_eq!(
        file.dataset("labels").unwrap().read_string().unwrap(),
        generation(9)
    );
}

/// The crash half of the same defect: a builder whose staged element data is
/// replaced by *fewer* bytes than the staging describes (issue #321).
///
/// `patch_vl_refs_masked` writes eight bytes at each staged offset without
/// checking them against the buffer, so a surviving staging over a shorter
/// replacement indexed out of bounds. On the whole-file writer that was a panic
/// out of `FileBuilder::write` in every released version, reachable from a
/// public builder with no unsafe and no malformed file involved.
///
/// Dropping the staging with the bytes it described removes the crash and the
/// silent half together, which is why this asserts a plain successful write.
#[test]
fn replacing_staged_data_with_fewer_bytes_does_not_panic() {
    let path = temp_path("hdf5_pure_edit_short_raw_after_staging.h5");
    let dt = {
        let seed = temp_path("hdf5_pure_edit_short_raw_seed.h5");
        let mut t = FileBuilder::new();
        t.create_dataset("x").with_vlen_strings(&["a", "b"]);
        t.write(&seed).unwrap();
        let f = File::open(&seed).unwrap();
        f.dataset("x").unwrap().datatype().unwrap()
    };

    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_shape(&[1])
        .with_vlen_strings(&["aa", "bb"])
        .with_raw_data(dt, vec![0u8; 16], 1);
    b.write(&path)
        .expect("a shorter replacement must not panic");

    // One element, and it is the raw bytes as given: an all-zero reference,
    // which is the null every reader answers for it.
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_string().unwrap(), vec![""]);
}

/// The object-reference half of the same builder invariant (issue #321).
///
/// `with_path_references` stages placeholder element bytes and a list of
/// targets to patch into them. Replacing the bytes must drop that list too:
/// patching is gated on the list being present, never on the datatype, so a
/// surviving list writes resolved object addresses over whatever the second
/// call supplied. Here that is ordinary `u64` data, and the file came back
/// holding the target's header address twice instead of the values asked for.
///
/// Its own crash half is quieter than the variable-length one's:
/// `write_reference_address` only `debug_assert!`s that the slot fits, so a
/// shorter replacement is a panic in a debug build and a silent skip in a
/// release one.
#[test]
fn replacing_a_builders_data_drops_the_reference_targets_that_described_it() {
    let path = temp_path("hdf5_pure_edit_raw_data_drops_reference_targets.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("target").with_i32_data(&[7]);
    b.create_dataset("d")
        .with_path_references(&["target", "target"])
        .with_u64_data(&[0xAAAA_AAAA, 0xBBBB_BBBB]);
    b.write(&path).unwrap();

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_u64().unwrap(),
        vec![0xAAAA_AAAAu64, 0xBBBB_BBBB],
        "the dropped reference targets were patched over the staged values"
    );
}

/// The provenance screen has two doors for raw element bytes, and this is the
/// one reached by *overwriting* rather than creating (issue #321).
///
/// `invalidate_heap_provenance` screens `staged.writes` and `staged.datasets`
/// alike, because either can carry element bytes the engine does not interpret.
/// A dataset overwritten with references lifted out of another one aliases its
/// collections just as a newly created one does, and the record has to be given
/// up for both.
#[test]
fn a_raw_bytes_overwrite_stops_a_vlen_overwrite_from_reclaiming_its_collection() {
    let path = temp_path("hdf5_pure_edit_raw_overwrite_blocks_reclaim.h5");
    let generation =
        |n: u32| -> Vec<String> { (0..2).map(|i| format!("gen-{n:02}-element-{i}")).collect() };

    let mut b = FileBuilder::new();
    b.create_dataset("labels")
        .with_vlen_strings(&generation(0).iter().map(String::as_str).collect::<Vec<_>>());
    b.create_dataset("alias")
        .with_vlen_strings(&["placeholder-x", "placeholder-y"]);
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        let overwrite = |n: u32| {
            let data = generation(n);
            session
                .dataset("labels")
                .unwrap()
                .write_staged(|b| {
                    b.with_vlen_strings(&data.iter().map(String::as_str).collect::<Vec<_>>());
                })
                .unwrap();
            session.commit().unwrap();
        };

        // Round one records the collections it places for "labels".
        overwrite(1);

        // Overwrite a *different* dataset with "labels"'s element references.
        let (dt, raw) = {
            let ds = session.dataset("labels").unwrap();
            (ds.datatype().unwrap(), ds.read_raw().unwrap())
        };
        session
            .dataset("alias")
            .unwrap()
            .write_staged(|b| {
                b.with_raw_data(dt, raw, 2);
            })
            .unwrap();
        session.commit().unwrap();

        for n in 2..8 {
            overwrite(n);
        }
    }

    let file = File::open(&path).unwrap();
    let alias = file.dataset("alias").unwrap().read_string();
    assert!(
        alias.as_ref().is_ok_and(|got| *got == generation(1)),
        "the aliased strings were freed out from under the overwritten dataset \
         and written over: {alias:?}"
    );
}

// ---------------------------------------------------------------------------
// Staged objects are addressable before the commit (issue #392)
// ---------------------------------------------------------------------------

#[test]
fn a_staged_subtree_is_addressable_before_the_commit() {
    let path = temp_path("hdf5_pure_staged_visible_subtree.h5");
    write_starter(&path);

    let session = File::open_rw(&path).unwrap();
    // The issue's reproducer: describe a nested schema, then bind handles to it
    // without a commit in between.
    let epoch = session
        .root()
        .create_group_with("epoch", |g| {
            g.create_dataset("sys_time", |b| {
                b.with_u64_data(&[])
                    .with_shape(&[0])
                    .with_maxshape(&[u64::MAX])
                    .with_chunks(&[4]);
            });
        })
        .unwrap();

    // Reachable by name from the file, from the root, and from the returned
    // handle — all before the commit.
    assert!(session.group("epoch").is_ok());
    assert!(session.dataset("epoch/sys_time").is_ok());
    assert!(session.root().group("epoch").is_ok());
    let mut col = epoch.dataset("sys_time").unwrap();

    // What the builder settled is answered from the staged record.
    assert_eq!(col.shape().unwrap(), vec![0]);
    assert_eq!(col.maxshape().unwrap(), Some(vec![u64::MAX]));
    assert_eq!(col.dtype().unwrap(), DType::U64);
    assert!(col.is_chunked());
    assert!(col.filters().is_empty());

    // An append onto a dataset that is not written yet folds into the pending
    // creation, and the shape follows.
    col.append_staged(|a| {
        a.append_u64(&[1, 2, 3]);
    })
    .unwrap();
    assert_eq!(col.shape().unwrap(), vec![3]);

    session.commit().unwrap();

    // The same handles now read the objects in the file.
    assert_eq!(col.read_u64().unwrap(), vec![1, 2, 3]);
    assert_eq!(epoch.datasets().unwrap(), vec!["sys_time".to_string()]);
    // The handles keep the session's exclusive lock alive; drop them before
    // reopening, since that lock is mandatory on Windows.
    drop(col);
    drop(epoch);
    drop(session);

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("epoch/sys_time").unwrap().read_u64().unwrap(),
        vec![1, 2, 3]
    );
}

#[test]
fn a_staged_dataset_handle_is_an_ordinary_dataset_handle() {
    // The composability the issue asks for: code that takes `&Dataset` accepts
    // one onto a dataset that is not committed yet.
    fn width(ds: &hdf5_pure::Dataset) -> u64 {
        ds.shape().unwrap().first().copied().unwrap_or(0)
    }
    fn count(group: &hdf5_pure::Group) -> usize {
        group.datasets().unwrap().len()
    }

    let path = temp_path("hdf5_pure_staged_visible_borrowed.h5");
    write_starter(&path);

    let session = File::open_rw(&path).unwrap();
    let grp = session.root().create_group("run").unwrap();
    let ds = grp
        .create_dataset("col", |b| {
            b.with_i32_data(&[1, 2, 3, 4]);
        })
        .unwrap();
    assert_eq!(width(&ds), 4);
    assert_eq!(count(&grp), 1);
    session.commit().unwrap();
    assert_eq!(width(&ds), 4);
    assert_eq!(count(&grp), 1);
}

#[test]
fn reading_a_staged_object_reports_that_it_is_not_committed() {
    let path = temp_path("hdf5_pure_staged_visible_reads.h5");
    write_starter(&path);

    let session = File::open_rw(&path).unwrap();
    let grp = session.root().create_group("run").unwrap();
    let mut col = grp
        .create_dataset("col", |b| {
            b.with_i32_data(&[7, 8])
                .with_maxshape(&[u64::MAX])
                .with_chunks(&[2]);
        })
        .unwrap();

    // A whole-dataset read must not answer fill values for elements that are
    // only staged.
    assert!(
        matches!(col.read_i32(), Err(Error::NotCommitted(p)) if p == "run/col"),
        "a staged dataset must not read as fill values"
    );
    assert!(matches!(col.read_raw(), Err(Error::NotCommitted(_))));
    assert!(matches!(col.attrs(), Err(Error::NotCommitted(_))));
    assert!(matches!(grp.attrs(), Err(Error::NotCommitted(_))));

    // In-place growth has no bytes to grow.
    assert!(matches!(col.append(&[9i32]), Err(Error::NotCommitted(_))));
    assert!(matches!(
        col.append_raw(&[0u8; 4]),
        Err(Error::NotCommitted(_))
    ));
    assert!(matches!(
        col.buffered_appender().map(|_| ()),
        Err(Error::NotCommitted(_))
    ));

    // So do the staged edits that rewrite an object the file already holds.
    assert!(matches!(col.write(&[1i32, 2]), Err(Error::NotCommitted(_))));
    assert!(matches!(
        col.write_staged(|b| {
            b.with_i32_data(&[1, 2]);
        }),
        Err(Error::NotCommitted(_))
    ));
    assert!(matches!(
        col.set_attr("units", AttrValue::I64(1)),
        Err(Error::NotCommitted(_))
    ));

    // A staged group takes attributes, which is what `create_group_with`'s
    // closure was the only way to do.
    grp.set_attr("kind", AttrValue::I64(3)).unwrap();
    session.commit().unwrap();
    assert_eq!(col.read_i32().unwrap(), vec![7, 8]);
    assert_eq!(grp.attrs().unwrap()["kind"], AttrValue::I64(3));
}

#[test]
fn a_replaced_path_resolves_to_the_object_replacing_it() {
    let path = temp_path("hdf5_pure_staged_visible_replace.h5");
    write_starter(&path);

    let session = File::open_rw(&path).unwrap();
    session.root().delete("original").unwrap();
    session
        .root()
        .create_dataset("original", |b| {
            b.with_i32_data(&[5, 6]);
        })
        .unwrap();

    // Before the commit the path already names the replacement, not the object
    // being removed — and reading it says so rather than answering with the old
    // dataset's values.
    let replacement = session.dataset("original").unwrap();
    assert_eq!(replacement.dtype().unwrap(), DType::I32);
    assert_eq!(replacement.shape().unwrap(), vec![2]);
    assert!(matches!(
        replacement.read_i32(),
        Err(Error::NotCommitted(_))
    ));
    // Listed once, not twice.
    assert_eq!(
        session.root().datasets().unwrap(),
        vec!["original".to_string()]
    );

    session.commit().unwrap();
    assert_eq!(replacement.read_i32().unwrap(), vec![5, 6]);
}

#[test]
fn listings_merge_staged_children_without_duplicating_them() {
    let path = temp_path("hdf5_pure_staged_visible_listings.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("kept").with_i32_data(&[1]);
    let mut shed = b.create_group("shed");
    shed.create_dataset("x").with_i32_data(&[2]);
    b.add_group(shed.finish());
    b.write(&path).unwrap();

    let session = File::open_rw(&path).unwrap();
    let root = session.root();
    root.create_dataset("added", |b| {
        b.with_i32_data(&[3]);
    })
    .unwrap();
    root.create_group_with("nested", |g| {
        g.create_dataset("leaf", |b| {
            b.with_i32_data(&[4]);
        });
    })
    .unwrap();
    // A replacement: the name is on disk and staged at once.
    root.delete("kept").unwrap();
    root.create_dataset("kept", |b| {
        b.with_i32_data(&[5]);
    })
    .unwrap();

    let mut datasets = root.datasets().unwrap();
    datasets.sort();
    assert_eq!(datasets, vec!["added".to_string(), "kept".to_string()]);
    let mut groups = root.groups().unwrap();
    groups.sort();
    assert_eq!(groups, vec!["nested".to_string(), "shed".to_string()]);

    // The iterating forms agree, and hand back usable handles.
    let mut walked: Vec<String> = root.iter_datasets().unwrap().map(|(n, _)| n).collect();
    walked.sort();
    assert_eq!(walked, datasets);
    let mut walked_groups: Vec<String> = root.iter_groups().unwrap().map(|(n, _)| n).collect();
    walked_groups.sort();
    assert_eq!(walked_groups, groups);

    // A staged group lists its staged members, having no on-disk links at all.
    let nested = root.group("nested").unwrap();
    assert_eq!(nested.datasets().unwrap(), vec!["leaf".to_string()]);
    assert!(nested.groups().unwrap().is_empty());

    session.commit().unwrap();
    let mut after = root.datasets().unwrap();
    after.sort();
    assert_eq!(after, datasets);
    assert_eq!(
        session.dataset("kept").unwrap().read_i32().unwrap(),
        vec![5]
    );
}

#[test]
fn a_refused_commit_leaves_the_staged_handle_pending() {
    let path = temp_path("hdf5_pure_staged_visible_refused.h5");
    write_starter(&path);

    let session = File::open_rw(&path).unwrap();
    let ds = session
        .root()
        .create_dataset("added", |b| {
            b.with_i32_data(&[1, 2]);
        })
        .unwrap();
    // A deletion of a path that does not exist refuses the whole commit, so the
    // addition beside it stays staged (issue #316).
    session.root().delete("no_such_object").unwrap();
    assert!(session.commit().is_err());
    assert!(
        matches!(ds.read_i32(), Err(Error::NotCommitted(p)) if p == "added"),
        "a refused commit leaves the object staged"
    );
    assert_eq!(ds.shape().unwrap(), vec![2]);

    // Dropping the `File` does not end the session while a handle onto it is
    // alive, so the object stays staged and the handle keeps saying so. The
    // file itself never received it.
    drop(session);
    assert!(matches!(ds.read_i32(), Err(Error::NotCommitted(_))));
    drop(ds);
    let reopened = File::open(&path).unwrap();
    assert!(matches!(
        reopened.dataset("added"),
        Err(Error::Format(FormatError::PathNotFound(_)))
    ));
}

#[test]
fn a_lookup_of_a_staged_object_of_the_other_kind_is_refused() {
    let path = temp_path("hdf5_pure_staged_visible_kinds.h5");
    write_starter(&path);

    let session = File::open_rw(&path).unwrap();
    session.root().create_group("grp").unwrap();
    session
        .root()
        .create_dataset("set", |b| {
            b.with_i32_data(&[1]);
        })
        .unwrap();

    assert!(matches!(session.dataset("grp"), Err(Error::NotADataset(_))));
    assert!(matches!(session.group("set"), Err(Error::NotAGroup(_))));
    assert!(matches!(
        session.root().dataset("grp"),
        Err(Error::NotADataset(_))
    ));
    assert!(matches!(
        session.root().group("set"),
        Err(Error::NotAGroup(_))
    ));
}

#[test]
fn appending_to_a_staged_dataset_of_the_wrong_type_is_refused() {
    let path = temp_path("hdf5_pure_staged_visible_append_type.h5");
    write_starter(&path);

    let session = File::open_rw(&path).unwrap();
    let mut col = session
        .root()
        .create_dataset("col", |b| {
            b.with_i32_data(&[1, 2]);
        })
        .unwrap();
    assert!(matches!(
        col.append_staged(|a| {
            a.append_f64(&[1.0]);
        }),
        Err(Error::AppendUnsupported(_))
    ));
    // The refusal staged nothing: the dataset is the one the builder described.
    assert_eq!(col.shape().unwrap(), vec![2]);

    // A contiguous staged dataset grows all the same — this is the builder's
    // data being extended, not a dataset in the file being appended to.
    col.append_staged(|a| {
        a.append_i32(&[3]);
    })
    .unwrap();
    session.commit().unwrap();
    assert_eq!(col.read_i32().unwrap(), vec![1, 2, 3]);
}

#[test]
fn appending_through_a_handle_onto_a_replaced_dataset_is_refused() {
    let path = temp_path("hdf5_pure_staged_visible_replace_append.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("x")
        .with_i32_data(&[1, 2, 3])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4]);
    b.write(&path).unwrap();

    let session = File::open_rw(&path).unwrap();
    // A handle onto the object in the file, taken before it is replaced.
    let mut doomed = session.dataset("x").unwrap();
    session.root().delete("x").unwrap();
    session
        .root()
        .create_dataset("x", |b| {
            b.with_i32_data(&[100])
                .with_maxshape(&[u64::MAX])
                .with_chunks(&[4]);
        })
        .unwrap();

    // The elements were meant for the dataset this handle names, which the
    // commit is removing. Growing the replacement under it would move them to
    // another object without saying so.
    let err = doomed
        .append_staged(|a| {
            a.append_i32(&[200]);
        })
        .unwrap_err();
    assert!(
        matches!(&err, Error::EditUnsupported(m) if m.contains("could only grow the object being replaced")),
        "unexpected error: {err:?}"
    );

    session.commit().unwrap();
    // Handles keep the session's exclusive lock alive, and that lock is
    // mandatory on Windows, so they go before the file is opened again.
    drop(doomed);
    drop(session);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("x").unwrap().read_i32().unwrap(), vec![100]);
}

#[test]
fn a_creation_colliding_with_a_surviving_link_is_refused_at_the_call() {
    let path = temp_path("hdf5_pure_staged_visible_collision.h5");
    write_starter(&path);

    let session = File::open_rw(&path).unwrap();
    // No `delete` beside it, so this creation collides with a link the commit
    // keeps — and `create_dataset` hands back a handle onto what it stages, so
    // the refusal has to come from the call rather than from the commit. A
    // handle onto a creation that never lands would answer for the file's own
    // `original` while claiming to address the new dataset.
    let err = session
        .root()
        .create_dataset("original", |b| {
            b.with_i32_data(&[9]);
        })
        .unwrap_err();
    assert!(err.to_string().contains("already exists"), "got: {err}");
    // The same rule when the creation would change the object's kind, which a
    // listing would otherwise report as a dataset becoming a group.
    let err = session.root().create_group("original").unwrap_err();
    assert!(err.to_string().contains("already exists"), "got: {err}");

    // Nothing was staged, so `original` still means the object in the file and
    // there is nothing left for the commit to refuse.
    assert_eq!(
        session.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        session.root().datasets().unwrap(),
        vec!["original".to_string()]
    );
    assert!(session.root().groups().unwrap().is_empty());
    assert!(!session.has_staged_edits());
    session.commit().unwrap();
    assert_eq!(
        session.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn a_creation_under_a_deleted_group_does_not_hide_the_files_own_child() {
    // `delete("g")` and `create_dataset("g/other")` with nothing rebuilding `g`
    // is a batch the commit refuses, so until it does, `g`'s own children are
    // still what their names mean: the deletion hides nothing on its own
    // (issue #392).
    let path = temp_path("hdf5_pure_staged_prefix_delete.h5");
    {
        let mut b = FileBuilder::new();
        let mut g = b.create_group("g");
        g.create_dataset("inner").with_i32_data(&[1, 2]);
        b.add_group(g.finish());
        b.write(&path).unwrap();
    }

    let session = File::open_rw(&path).unwrap();
    session.root().delete("g").unwrap();
    // Deleting `g` does not hand `g/inner` over to a creation there: only a
    // commit that builds `g` again could, and this one does not. So the name is
    // still taken, and staging over it is the collision it looks like.
    let err = session
        .root()
        .create_dataset("g/inner", |b| {
            b.with_i32_data(&[9]);
        })
        .unwrap_err();
    assert!(err.to_string().contains("already exists"), "got: {err}");
    session
        .root()
        .create_dataset("g/other", |b| {
            b.with_i32_data(&[9]);
        })
        .unwrap();

    let g = session.group("g").unwrap();
    assert_eq!(
        g.dataset("inner").unwrap().read_i32().unwrap(),
        vec![1, 2],
        "a live, readable dataset must not be reported as staged"
    );
    assert!(
        g.datasets().unwrap().contains(&"inner".to_string()),
        "it must not vanish from its group's listing either"
    );
    // The batch itself is still the commit's to refuse.
    assert!(session.commit().is_err());
}

#[test]
fn the_root_group_cannot_be_deleted() {
    let path = temp_path("hdf5_pure_staged_delete_root.h5");
    write_starter(&path);

    let session = File::open_rw(&path).unwrap();
    for name in ["", "/"] {
        let err = session.root().delete(name).unwrap_err();
        assert!(err.to_string().contains("root group"), "got: {err}");
    }
    // Refused where it was asked for, so it cannot go on to make every staged
    // creation in the session look like a replacement of a file object.
    assert!(!session.has_staged_edits());
    session
        .root()
        .create_dataset("added", |b| {
            b.with_i32_data(&[1]);
        })
        .unwrap();
    assert_eq!(
        session.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    session.commit().unwrap();
}

#[test]
fn deleting_a_staged_object_withdraws_it() {
    let path = temp_path("hdf5_pure_staged_visible_withdraw.h5");
    write_starter(&path);
    let before = std::fs::read(&path).unwrap();

    let session = File::open_rw(&path).unwrap();
    let root = session.root();
    let ds = root
        .create_dataset("added", |b| {
            b.with_i32_data(&[1, 2]);
        })
        .unwrap();
    let grp = root
        .create_group_with("nested", |g| {
            g.create_dataset("leaf", |b| {
                b.with_i32_data(&[3]);
            });
        })
        .unwrap();
    grp.set_attr("kind", AttrValue::I64(1)).unwrap();

    root.delete("added").unwrap();
    root.delete("nested").unwrap();
    assert!(!session.has_staged_edits(), "the staging was withdrawn");

    // The handles name creations the session has dropped, and say exactly that
    // rather than reporting the path as merely absent: a handle born onto a
    // staged creation is never quietly retargeted at what the path means now.
    assert!(matches!(ds.read_i32(), Err(Error::StagingWithdrawn(p)) if p == "added"));
    assert!(matches!(grp.attrs(), Err(Error::StagingWithdrawn(p)) if p == "nested"));
    // Nor may one go on *staging* through the withdrawn path, which is how an
    // edit would end up landing under whatever the file holds there.
    assert!(matches!(
        grp.create_dataset("late", |b| {
            b.with_i32_data(&[1]);
        }),
        Err(Error::StagingWithdrawn(_))
    ));
    assert!(matches!(
        grp.set_attr("kind", AttrValue::I64(2)),
        Err(Error::StagingWithdrawn(_))
    ));
    assert!(matches!(
        grp.delete("leaf"),
        Err(Error::StagingWithdrawn(_))
    ));
    assert!(!session.has_staged_edits(), "and none of them recorded");
    // A fresh lookup of the same name is a different question, and answers the
    // way a lookup of anything absent does.
    assert!(matches!(
        session.dataset("added"),
        Err(Error::Format(FormatError::PathNotFound(_)))
    ));
    assert!(matches!(
        session.group("nested"),
        Err(Error::Format(FormatError::PathNotFound(_)))
    ));
    assert_eq!(
        session.root().datasets().unwrap(),
        vec!["original".to_string()]
    );

    // A commit of nothing leaves the file byte-identical.
    session.commit().unwrap();
    // Handles keep the session's exclusive lock alive, and that lock is
    // mandatory on Windows, so they go before the file is opened again.
    drop(ds);
    drop(grp);
    drop(root);
    drop(session);
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn deleting_a_staged_replacement_leaves_the_deletion_it_replaced() {
    let path = temp_path("hdf5_pure_staged_visible_withdraw_replacement.h5");
    write_starter(&path);

    let session = File::open_rw(&path).unwrap();
    let root = session.root();
    root.delete("original").unwrap();
    let mut replacement = root
        .create_dataset("original", |b| {
            b.with_i32_data(&[9]);
        })
        .unwrap();
    assert_eq!(replacement.shape().unwrap(), vec![1], "the replacement's");
    // Changing one's mind about the replacement leaves the deletion that was
    // asked for first, rather than a staged creation nothing can withdraw.
    root.delete("original").unwrap();
    // The handle was made onto that creation, and the file's own `original` is
    // the object this session is *deleting* — so the handle reports the
    // withdrawal rather than quietly answering, reading or editing for it
    // (issue #392).
    assert!(
        matches!(replacement.shape(), Err(Error::StagingWithdrawn(p)) if p == "original"),
        "got: {:?}",
        replacement.shape()
    );
    assert!(matches!(
        replacement.read_f64(),
        Err(Error::StagingWithdrawn(_))
    ));
    assert!(matches!(
        replacement.append_staged(|a| {
            a.append_i32(&[1]);
        }),
        Err(Error::StagingWithdrawn(_))
    ));
    assert!(matches!(
        replacement.write(&[1i32]),
        Err(Error::StagingWithdrawn(_))
    ));
    assert!(matches!(
        replacement.set_attr("kind", AttrValue::I64(1)),
        Err(Error::StagingWithdrawn(_))
    ));
    // A staged deletion hides nothing, so a fresh lookup still means the object
    // in the file until the commit removes it — the answer it gave before the
    // replacement was staged.
    assert_eq!(
        session.dataset("original").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    session.commit().unwrap();
    // Handles keep the session's exclusive lock alive, and that lock is
    // mandatory on Windows, so they go before the file is opened again.
    drop(replacement);
    drop(root);
    drop(session);

    let file = File::open(&path).unwrap();
    assert!(file.root().datasets().unwrap().is_empty());
}
