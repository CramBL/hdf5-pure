//! Object paths through the public entry points: every spelling of one path, a path relative to a
//! group, an absolute path from either, and the path each error reports the object by.
//!
//! `crates/crosscheck/tests/object_path.rs` runs the same spellings through the C library, which
//! is the oracle for the grammar.

use hdf5_pure::{Dataset, Error, File, FileBuilder, FormatError, Group};
use rstest::rstest;
use tempfile::tempdir;

/// Returns a file holding `/b`, `/a/b`, `/a/mytype` and `/a/inner`, where `/b` reads 2 and `/a/b`
/// reads 1, so that a path resolved from the root and the same path resolved from `a` reach
/// different data.
fn nested_file() -> Vec<u8> {
    let mut builder = FileBuilder::new();
    builder.create_dataset("b").with_i32_data(&[2]);
    let mut a = builder.create_group("a");
    a.create_dataset("b").with_i32_data(&[1]);
    a.commit_datatype("mytype", hdf5_pure::make_i32_type());
    let inner = a.create_group("inner");
    a.add_group(inner.finish());
    builder.add_group(a.finish());
    builder.finish().unwrap()
}

/// The two entry points that resolve a path from the root: `File::dataset`, and `Group::dataset`
/// on the root group, which agree on what a path identifies and on what an error reports.
#[derive(Clone, Copy, Debug)]
enum EntryPoint {
    File,
    Root,
}

impl EntryPoint {
    fn dataset(self, file: &File, path: &str) -> Result<Dataset, Error> {
        match self {
            EntryPoint::File => file.dataset(path),
            EntryPoint::Root => file.root().dataset(path),
        }
    }
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
fn every_spelling_of_a_path_names_one_dataset(
    #[case] spelling: &str,
    #[values(EntryPoint::File, EntryPoint::Root)] entry: EntryPoint,
) {
    let file = File::from_bytes(nested_file()).unwrap();
    assert_eq!(
        entry.dataset(&file, spelling).unwrap().read_i32().unwrap(),
        vec![1]
    );
}

#[test]
fn a_group_resolves_every_kind_of_object_below_it() {
    let file = File::from_bytes(nested_file()).unwrap();
    let root = file.root();

    assert_eq!(root.dataset("a/b").unwrap().read_i32().unwrap(), vec![1]);
    assert_eq!(
        root.group("a/inner").unwrap().groups().unwrap(),
        Vec::<String>::new()
    );
    assert_eq!(
        root.named_datatype("a/mytype").unwrap(),
        hdf5_pure::make_i32_type()
    );
    assert_eq!(root.named_datatype_references("a/mytype").unwrap(), 1);
}

#[rstest]
#[case("b")]
#[case("./b")]
#[case("b/")]
fn a_subgroup_resolves_a_path_relative_to_itself(#[case] spelling: &str) {
    let file = File::from_bytes(nested_file()).unwrap();
    let a = file.group("a").unwrap();
    assert_eq!(a.dataset(spelling).unwrap().read_i32().unwrap(), vec![1]);
}

#[rstest]
#[case("/b", 2)]
#[case("//b", 2)]
#[case("/a/b", 1)]
#[case("/./a/b/", 1)]
fn an_absolute_path_through_a_subgroup_starts_at_the_root(
    #[case] spelling: &str,
    #[case] value: i32,
) {
    let file = File::from_bytes(nested_file()).unwrap();
    let a = file.group("a").unwrap();
    assert_eq!(
        a.dataset(spelling).unwrap().read_i32().unwrap(),
        vec![value]
    );
}

#[test]
fn an_absolute_path_stages_a_creation_at_the_root() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("absolute.h5");
    let mut builder = FileBuilder::new();
    let a = builder.create_group("a");
    builder.add_group(a.finish());
    builder.write(&path).unwrap();

    let file = File::open_rw(&path).unwrap();
    let a = file.group("a").unwrap();
    a.create_dataset("/z", |b| {
        b.with_i32_data(&[7]);
    })
    .unwrap();
    file.commit().unwrap();

    assert_eq!(file.root().datasets().unwrap(), vec!["z"]);
    assert_eq!(a.datasets().unwrap(), Vec::<String>::new());
    assert_eq!(file.dataset("z").unwrap().read_i32().unwrap(), vec![7]);
}

#[rstest]
#[case("")]
#[case(".")]
#[case("./.")]
fn a_relative_path_that_spells_no_link_identifies_the_group_it_is_relative_to(
    #[case] spelling: &str,
) {
    let file = File::from_bytes(nested_file()).unwrap();

    let err = file.group("a").unwrap().dataset(spelling).unwrap_err();
    let Error::NotADataset(named) = &err else {
        panic!("expected NotADataset, got {err:?}");
    };
    assert_eq!(named, "a");

    assert_eq!(file.group(spelling).unwrap().datasets().unwrap(), vec!["b"]);
}

#[rstest]
#[case("/")]
#[case("//")]
#[case("/./")]
fn an_absolute_path_that_spells_no_link_identifies_the_root(#[case] spelling: &str) {
    let file = File::from_bytes(nested_file()).unwrap();

    let err = file.group("a").unwrap().dataset(spelling).unwrap_err();
    let Error::NotADataset(named) = &err else {
        panic!("expected NotADataset, got {err:?}");
    };
    assert!(named.is_empty(), "the root is identified by the empty path");

    assert_eq!(
        file.group("a")
            .unwrap()
            .group(spelling)
            .unwrap()
            .datasets()
            .unwrap(),
        vec!["b"]
    );
}

#[rstest]
#[case("absent", "absent")]
#[case("a/absent", "a/absent")]
#[case("/a//absent/", "a/absent")]
fn a_component_that_reaches_nothing_is_named_by_its_path(
    #[case] asked: &str,
    #[case] missing: &str,
    #[values(EntryPoint::File, EntryPoint::Root)] entry: EntryPoint,
) {
    let file = File::from_bytes(nested_file()).unwrap();
    let err = entry.dataset(&file, asked).unwrap_err();
    let Error::Format(FormatError::PathNotFound(named)) = &err else {
        panic!("expected PathNotFound, got {err:?}");
    };
    assert_eq!(named, missing);
}

#[rstest]
#[case("b/deeper", "b")]
#[case("a/b/deeper", "a/b")]
#[case("a/mytype/deeper", "a/mytype")]
fn an_object_that_stopped_a_walk_is_named_by_its_path(
    #[case] asked: &str,
    #[case] stopper: &str,
    #[values(EntryPoint::File, EntryPoint::Root)] entry: EntryPoint,
) {
    let file = File::from_bytes(nested_file()).unwrap();
    let err = entry.dataset(&file, asked).unwrap_err();
    let Error::NotAGroup(named) = &err else {
        panic!("expected NotAGroup, got {err:?}");
    };
    assert_eq!(named, stopper);
}

#[test]
fn a_subgroup_names_the_object_that_stopped_its_walk_from_the_root() {
    let file = File::from_bytes(nested_file()).unwrap();
    let err = file.group("a").unwrap().dataset("b/deeper").unwrap_err();
    let Error::NotAGroup(named) = &err else {
        panic!("expected NotAGroup, got {err:?}");
    };
    assert_eq!(named, "a/b");
}

#[test]
fn a_group_created_at_a_path_is_found_at_that_path() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("created.h5");
    let mut builder = FileBuilder::new();
    let x = builder.create_group("x");
    builder.add_group(x.finish());
    builder.write(&path).unwrap();

    let file = File::open_rw(&path).unwrap();
    let root = file.root();
    root.create_group("x/y").unwrap();
    file.commit().unwrap();
    assert_eq!(
        root.group("x/y").unwrap().datasets().unwrap(),
        Vec::<String>::new()
    );

    root.create_dataset("x//y/./z", |b| {
        b.with_i32_data(&[7]);
    })
    .unwrap();
    file.commit().unwrap();
    assert_eq!(root.dataset("x/y/z").unwrap().read_i32().unwrap(), vec![7]);
    assert_eq!(
        file.dataset("/x/y/z/").unwrap().read_i32().unwrap(),
        vec![7]
    );
}

#[test]
fn a_group_rejected_as_a_dataset_is_reported_the_same_way_before_and_after_a_commit() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("staged.h5");
    FileBuilder::new().write(&path).unwrap();

    let file = File::open_rw(&path).unwrap();
    let root = file.root();
    root.create_group("g").unwrap();

    let staged = file.dataset("/g/").unwrap_err();
    let Error::NotADataset(named) = &staged else {
        panic!("expected NotADataset, got {staged:?}");
    };
    assert_eq!(named, "g");

    file.commit().unwrap();
    let written = file.dataset("/g/").unwrap_err();
    let Error::NotADataset(named) = &written else {
        panic!("expected NotADataset, got {written:?}");
    };
    assert_eq!(named, "g");
}

/// The fixture's one link has an empty name, which no path spells.
#[test]
fn a_link_whose_name_is_no_path_component_is_listed_and_not_opened() {
    let bytes =
        std::fs::read("tests/data/fuzz/oom_chunked_string_huge_elem.h5").expect("read fixture");
    let file = File::from_bytes(bytes).unwrap();
    let root = file.root();

    assert_eq!(root.datasets().unwrap(), vec![""]);

    let err = root.dataset("").unwrap_err();
    let Error::NotADataset(named) = &err else {
        panic!("expected NotADataset, got {err:?}");
    };
    assert!(
        named.is_empty(),
        "the empty path names the root, got {named:?}"
    );

    let (name, dataset) = root.iter_datasets().unwrap().next().unwrap();
    assert_eq!(name, "");
    assert_eq!(dataset.shape().unwrap(), vec![50]);
}

/// The three writes a group stages for a child identified by path.
#[derive(Clone, Copy, Debug)]
enum StagedWrite {
    CreateDataset,
    CreateGroup,
    Delete,
}

impl StagedWrite {
    fn stage(self, group: &Group, path: &str) -> Result<(), Error> {
        match self {
            StagedWrite::CreateDataset => group
                .create_dataset(path, |b| {
                    b.with_i32_data(&[1]);
                })
                .map(|_| ()),
            StagedWrite::CreateGroup => group.create_group(path).map(|_| ()),
            StagedWrite::Delete => group.delete(path),
        }
    }
}

#[rstest]
#[case("")]
#[case(".")]
#[case("./.")]
fn a_path_that_spells_no_link_is_no_target_for_a_write(
    #[case] spelling: &str,
    #[values(
        StagedWrite::CreateDataset,
        StagedWrite::CreateGroup,
        StagedWrite::Delete
    )]
    write: StagedWrite,
) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("write.h5");
    let mut builder = FileBuilder::new();
    let mut a = builder.create_group("a");
    a.create_dataset("d").with_i32_data(&[1]);
    builder.add_group(a.finish());
    builder.write(&path).unwrap();

    let file = File::open_rw(&path).unwrap();
    let a = file.group("a").unwrap();

    let err = write.stage(&a, spelling).unwrap_err();
    let Error::EditUnsupported(reason) = &err else {
        panic!("expected EditUnsupported, got {err:?}");
    };
    assert_eq!(
        *reason,
        "a write needs a link name, and this path holds none"
    );

    file.commit().unwrap();
    assert_eq!(file.root().groups().unwrap(), vec!["a"]);
    assert_eq!(a.datasets().unwrap(), vec!["d"]);
}
