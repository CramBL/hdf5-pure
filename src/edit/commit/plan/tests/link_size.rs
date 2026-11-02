//! Staged-input ownership and I/O during rejection of oversized Link messages.

use std::path::Path;

use rstest::rstest;
use test_util::disk_log;

use super::super::CommitPreparation;
use super::super::PreparedCommit;
use crate::File;
use crate::FileBuilder;
use crate::edit::WriteEngine;
use crate::edit::dataset;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::object_path::LinkNameBuf;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::root_group(Destination::Group, "")]
#[case::nested_group(Destination::Group, "parent")]
#[case::root_dataset(Destination::Dataset, "")]
#[case::nested_dataset(Destination::Dataset, "parent")]
#[case::root_same_file_copy(Destination::SameFileCopy, "")]
#[case::nested_same_file_copy(Destination::SameFileCopy, "parent")]
#[case::root_cross_file_copy(Destination::CrossFileCopy, "")]
#[case::nested_cross_file_copy(Destination::CrossFileCopy, "parent")]
fn oversized_links_return_staged_inputs_without_placement(
    #[case] destination: Destination,
    #[case] parent: &str,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("destination.h5");
    let source_path = directory.path().join("source.h5");
    write_fixture(&path);
    std::fs::copy(&path, &source_path).unwrap();
    let source = File::open(&source_path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .create_group(&ObjectPathBuf::parse("valid_group"))
        .unwrap();
    engine
        .stage_created_dataset(
            &ObjectPathBuf::parse("valid_dataset"),
            super::i32_dataset(&[23]),
        )
        .unwrap();
    engine
        .copy(
            &ObjectPathBuf::parse("template"),
            &ObjectPathBuf::parse("valid_copy"),
        )
        .unwrap();
    engine
        .copy_from(
            &source,
            &ObjectPathBuf::parse("template"),
            &ObjectPathBuf::parse("valid_cross_copy"),
        )
        .unwrap();
    destination.stage(
        &mut engine,
        &source,
        ObjectPathBuf::parse(parent),
        LinkNameBuf::new(&"x".repeat(65_532)).unwrap(),
    );
    let before = engine.disk_bytes().unwrap();
    let root = engine.context.superblock().clone();
    let length = engine.image.len();
    let free = engine
        .space
        .admitted_snapshot()
        .sections()
        .collect::<Vec<_>>();
    let staged = std::mem::take(&mut engine.staged);
    let snapshot = staged.snapshot();
    let datasets = staged.datasets().as_ptr();
    let groups = staged.groups().as_ptr();
    let copies = staged.copies().as_ptr();
    let cross_copies = staged.cross_copies().as_ptr();
    engine.begin_commit(&staged);
    disk_log::start();
    let preparation = PreparedCommit::prepare(&mut engine, staged);
    let io = disk_log::finish();
    let CommitPreparation::Rejected { staged, error } = preparation else {
        panic!("expected preflight to reject the oversized link");
    };
    assert!(
        matches!(
            error,
            Error::EditUnsupported(
                "a new hard link message exceeds the object-header message size limit"
            )
        ),
        "{error:?}"
    );
    assert!(io.events.is_empty(), "{io:?}");
    assert_eq!(engine.disk_bytes().unwrap(), before);
    assert_eq!(engine.image.len(), length);
    assert_eq!(engine.context.superblock(), &root);
    assert_eq!(
        engine
            .space
            .admitted_snapshot()
            .sections()
            .collect::<Vec<_>>(),
        free
    );
    assert!(staged.snapshot() == snapshot);
    assert_eq!(staged.datasets().as_ptr(), datasets);
    assert_eq!(staged.groups().as_ptr(), groups);
    assert_eq!(staged.copies().as_ptr(), copies);
    assert_eq!(staged.cross_copies().as_ptr(), cross_copies);
    drop(staged);
    drop(engine);
    drop(source);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().groups().unwrap(), ["parent", "template"]);
    assert_eq!(file.root().datasets().unwrap(), ["original"]);
    assert_eq!(
        file.dataset("original").unwrap().read_i32().unwrap(),
        [7, 11]
    );
    assert_eq!(
        file.dataset("parent/nested").unwrap().read_i32().unwrap(),
        [13]
    );
    assert_eq!(
        file.dataset("template/payload")
            .unwrap()
            .read_i32()
            .unwrap(),
        [17, 19]
    );
    drop(file);
    directory.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
enum Destination {
    CrossFileCopy,
    Dataset,
    Group,
    SameFileCopy,
}

impl Destination {
    /// Stages a destination for commit preflight.
    ///
    /// Assigns dataset names after flattening to exercise preflight independently of the
    /// dataset-name staging limit.
    fn stage(
        self,
        engine: &mut WriteEngine,
        source: &File,
        parent: ObjectPathBuf,
        name: LinkNameBuf,
    ) {
        let destination_path = parent.join(&name);
        match self {
            Destination::Group => {
                engine.create_group(&destination_path).unwrap();
                engine
                    .stage_created_dataset(
                        &destination_path.join(&LinkNameBuf::new("child").unwrap()),
                        super::i32_dataset(&[29]),
                    )
                    .unwrap();
            }
            Destination::Dataset => {
                let mut prepared = dataset::flatten_dataset(
                    super::i32_dataset(&[31]),
                    LinkNameBuf::new("short").unwrap(),
                )
                .unwrap();
                prepared.name = name;
                engine.staged.push_dataset(parent, prepared);
            }
            Destination::SameFileCopy => engine
                .copy(&ObjectPathBuf::parse("template"), &destination_path)
                .unwrap(),
            Destination::CrossFileCopy => engine
                .copy_from(source, &ObjectPathBuf::parse("template"), &destination_path)
                .unwrap(),
        }
    }
}

fn write_fixture(path: &Path) {
    let mut builder = FileBuilder::new();
    builder.create_dataset("original").with_i32_data(&[7, 11]);
    let mut group = builder.create_group("parent");
    group.create_dataset("nested").with_i32_data(&[13]);
    builder.add_group(group.finish());
    let mut template = builder.create_group("template");
    template.create_dataset("payload").with_i32_data(&[17, 19]);
    builder.add_group(template.finish());
    builder.write(path).unwrap();
}
