use std::collections::HashMap;
use std::path::Path;

use rstest::rstest;

use hdf5_pure::AttrValue;
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

#[rstest]
#[case::group(Destination::Group)]
#[case::same_file_copy(Destination::SameFileCopy)]
#[case::cross_file_copy(Destination::CrossFileCopy)]
fn oversized_destinations_reject_every_staged_insertion(
    #[case] destination: Destination,
    #[values("", "parent")] parent: &str,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("destination.h5");
    let source_path = directory.path().join("source.h5");
    write_destination_fixture(&path);
    std::fs::copy(&path, &source_path).unwrap();
    let before = std::fs::read(&path).unwrap();
    let source = File::open(&source_path).unwrap();
    let file = File::open_rw(&path).unwrap();
    file.root().create_group("valid_group").unwrap();
    file.root()
        .create_dataset("valid_dataset", |dataset| {
            dataset.with_i32_data(&[41]);
        })
        .unwrap();
    let name = "x".repeat(65_532);
    let path_name = destination_path(parent, &name);
    match destination {
        Destination::Group => {
            file.group(parent)
                .unwrap()
                .create_group(&name)
                .unwrap()
                .create_dataset("child", |dataset| {
                    dataset.with_i32_data(&[43]);
                })
                .unwrap();
        }
        Destination::SameFileCopy => file.copy("template", &path_name).unwrap(),
        Destination::CrossFileCopy => file.copy_from(&source, "template", &path_name).unwrap(),
    }
    assert_link_size_error(file.commit().unwrap_err());
    drop(file);
    drop(source);
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().groups().unwrap(), ["parent", "template"]);
    assert_eq!(file.root().datasets().unwrap(), ["original"]);
    assert_eq!(
        file.group("parent").unwrap().datasets().unwrap(),
        ["nested"]
    );
    assert_destination_values(&file);
    drop(file);
    directory.close().unwrap();
}

#[rstest]
#[case::root("")]
#[case::nested("parent")]
fn oversized_dataset_staging_preserves_accepted_names_and_pending_edits(#[case] parent: &str) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("dataset.h5");
    write_destination_fixture(&path);
    let file = File::open_rw(&path).unwrap();
    file.root().create_group("retained").unwrap();
    let group = file.group(parent).unwrap();
    let accepted = "a".repeat(65_515);
    group
        .create_dataset(&accepted, |dataset| {
            dataset.with_i32_data(&[47]);
        })
        .unwrap();
    let error = group
        .create_dataset(&"x".repeat(65_532), |dataset| {
            dataset.with_i32_data(&[53]);
        })
        .unwrap_err();
    assert!(
        matches!(
            error,
            Error::EditUnsupported("dataset name is too long to encode as a link message")
        ),
        "{error:?}"
    );
    group
        .create_dataset("short", |dataset| {
            dataset.with_i32_data(&[59]);
        })
        .unwrap();
    file.commit().unwrap();
    drop(group);
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset(&destination_path(parent, &accepted))
            .unwrap()
            .read_i32()
            .unwrap(),
        [47]
    );
    assert_eq!(
        file.dataset(&destination_path(parent, "short"))
            .unwrap()
            .read_i32()
            .unwrap(),
        [59]
    );
    let mut groups = file.root().groups().unwrap();
    groups.sort();
    assert_eq!(groups, ["parent", "retained", "template"]);
    let expected = if parent.is_empty() {
        vec!["original".to_string(), accepted, "short".into()]
    } else {
        vec!["nested".to_string(), accepted, "short".into()]
    };
    assert_eq!(file.group(parent).unwrap().datasets().unwrap(), expected);
    assert_destination_values(&file);
    drop(file);
    directory.close().unwrap();
}

fn write_destination_fixture(path: &Path) {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("original")
        .with_i32_data(&[7, 11, 13]);
    let mut parent = builder.create_group("parent");
    parent.set_attr("kept", AttrValue::I64(37));
    parent.create_dataset("nested").with_i32_data(&[17, 19]);
    builder.add_group(parent.finish());
    let mut template = builder.create_group("template");
    template
        .create_dataset("payload")
        .with_i32_data(&[23, 29, 31]);
    builder.add_group(template.finish());
    builder.write(path).unwrap();
}

fn assert_destination_values(file: &File) {
    assert_eq!(
        file.dataset("original").unwrap().read_i32().unwrap(),
        [7, 11, 13]
    );
    assert_eq!(
        file.dataset("parent/nested").unwrap().read_i32().unwrap(),
        [17, 19]
    );
    assert_eq!(
        file.dataset("template/payload")
            .unwrap()
            .read_i32()
            .unwrap(),
        [23, 29, 31]
    );
    assert_eq!(
        file.group("parent").unwrap().attrs().unwrap(),
        HashMap::from([("kept".into(), AttrValue::I64(37))])
    );
}

fn destination_path(parent: &str, name: &str) -> String {
    if parent.is_empty() {
        name.to_string()
    } else {
        format!("{parent}/{name}")
    }
}

fn assert_link_size_error(error: Error) {
    assert!(
        matches!(
            error,
            Error::EditUnsupported(
                "a new hard link message exceeds the object-header message size limit"
            )
        ),
        "{error:?}"
    );
}

#[derive(Clone, Copy, Debug)]
enum Destination {
    CrossFileCopy,
    Group,
    SameFileCopy,
}
