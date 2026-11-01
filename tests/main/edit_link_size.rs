use std::path::Path;

use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;

#[test]
fn oversized_group_link_rejection_leaves_the_published_file_unchanged() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("oversized-link.h5");
    write_original(&path);
    let before = std::fs::read(&path).unwrap();
    let (file, _) = reject_oversized_group_link(&path);
    drop(file);
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_published_data(&path, &[]);
    directory.close().unwrap();
}

#[test]
fn oversized_group_link_rejection_retains_staged_edits_for_retry() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("oversized-link.h5");
    write_original(&path);
    let (file, name) = reject_oversized_group_link(&path);
    file.root().delete(&name).unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    assert_published_data(&path, &["valid"]);
    directory.close().unwrap();
}

fn write_original(path: &Path) {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("original")
        .with_i32_data(&[7, 11, 13]);
    builder.write(path).unwrap();
}

fn reject_oversized_group_link(path: &Path) -> (File, String) {
    let file = File::open_rw(path).unwrap();
    let root = file.root();
    root.create_group("valid").unwrap();
    let name = "x".repeat(65_532);
    root.create_group(&name).unwrap();
    let error = file.commit().unwrap_err();
    assert!(
        matches!(
            error,
            Error::EditUnsupported(
                "a new hard link message exceeds the object-header message size limit"
            )
        ),
        "{error:?}"
    );
    drop(root);
    (file, name)
}

fn assert_published_data(path: &Path, groups: &[&str]) {
    let reopened = File::open(path).unwrap();
    assert_eq!(reopened.root().groups().unwrap(), groups);
    assert_eq!(reopened.root().datasets().unwrap(), ["original"]);
    assert_eq!(
        reopened.dataset("original").unwrap().read_i32().unwrap(),
        [7, 11, 13]
    );
    reopened.close().unwrap();
}
