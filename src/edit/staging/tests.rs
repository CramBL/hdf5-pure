use std::fs;
use std::path::Path;

use rstest::rstest;

use super::StagedKind;
use crate::dataspace::MaxExtent;
use crate::edit::{AppendBuilder, WriteEngine};
use crate::error::{Error, FormatError};
use crate::file_lock::FileLocking;
use crate::object_path::ObjectPathBuf;
use crate::reader::File;
use crate::type_builders::{AttrValue, DatasetBuilder};
use crate::writer::FileBuilder;

#[test]
fn failed_batch_restores_every_staged_operation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edits.h5");
    let source_path = dir.path().join("source.h5");
    let mut engine = open_session(&path);
    let source = source_file(&source_path);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("retained"), dataset(&[8]))
        .unwrap();
    let mark = engine.staged.mark();
    let generation = engine.staged_generation();

    let err = engine
        .stage_atomically(|session| {
            session.stage_operations(&source)?;
            Err::<(), Error>(Error::EditUnsupported("batch failed"))
        })
        .unwrap_err();
    assert!(
        matches!(err, Error::EditUnsupported("batch failed")),
        "{err:?}"
    );
    assert_eq!(engine.staged.mark(), mark);
    assert_eq!(engine.staged_generation(), generation);
    assert_eq!(
        engine
            .staged_object(&ObjectPathBuf::parse("retained"))
            .unwrap()
            .kind,
        StagedKind::Dataset
    );
    assert!(
        engine
            .staged_object(&ObjectPathBuf::parse("created"))
            .is_none()
    );
    assert!(
        engine
            .staged_object(&ObjectPathBuf::parse("group"))
            .is_none()
    );
    assert_eq!(
        engine
            .staged_children(&ObjectPathBuf::root())
            .into_iter()
            .map(|child| (child.name, child.kind, child.replaces_link))
            .collect::<Vec<_>>(),
        vec![("retained".to_string(), StagedKind::Dataset, false)]
    );

    engine.stage_operations(&source).unwrap();
    engine.commit().unwrap();
    assert!(engine.staged.is_empty());
    drop(engine);
    assert_contents(&path);
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("retained")
            .unwrap()
            .read_i32()
            .unwrap(),
        vec![8]
    );
}

#[test]
fn rejected_commit_restores_every_staged_operation_for_retry() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edits.h5");
    let source_path = dir.path().join("source.h5");
    let mut engine = open_session(&path);
    let source = source_file(&source_path);
    engine.stage_operations(&source).unwrap();
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("missing/pending"), dataset(&[9]))
        .unwrap();
    let mark = engine.staged.mark();
    let generation = engine.staged_generation();
    let before = engine.page_publication_snapshot().unwrap();

    let err = engine.commit().unwrap_err();
    let Error::Format(FormatError::PathNotFound(missing)) = &err else {
        panic!("expected PathNotFound, got {err:?}");
    };
    assert_eq!(missing, "missing");
    assert_eq!(engine.page_publication_snapshot().unwrap(), before);
    assert_eq!(
        fs::metadata(&path).unwrap().len(),
        u64::try_from(before.len()).unwrap()
    );
    assert_eq!(engine.staged.mark(), mark);
    assert_eq!(engine.staged_generation(), generation);
    assert_eq!(
        engine
            .staged_dataset_meta(&ObjectPathBuf::parse("created"))
            .unwrap()
            .dimensions,
        vec![1]
    );
    assert_eq!(
        engine
            .staged_object(&ObjectPathBuf::parse("group"))
            .unwrap()
            .kind,
        StagedKind::Group
    );
    assert_eq!(
        engine
            .staged_children(&ObjectPathBuf::root())
            .into_iter()
            .map(|child| child.name)
            .collect::<Vec<_>>(),
        vec!["group", "created"]
    );

    engine
        .delete(&ObjectPathBuf::parse("missing/pending"))
        .unwrap();
    engine.commit().unwrap();
    assert!(engine.staged.is_empty());
    drop(engine);
    assert_contents(&path);
}

#[rstest]
#[case::retain_outer(true)]
#[case::rollback_outer(false)]
fn nested_batches_restore_indexes_and_keep_the_outer_guard(#[case] retain_outer: bool) {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = open_session(&dir.path().join("edits.h5"));
    let outer_path = ObjectPathBuf::parse("outer");
    let inner_path = ObjectPathBuf::parse("inner");
    let result = engine.stage_atomically(|outer| {
        outer.create_group(&outer_path)?;
        let err = outer
            .stage_atomically(|inner| {
                inner.create_group(&inner_path)?;
                inner
                    .stage_created_dataset(&ObjectPathBuf::parse("inner/pending"), dataset(&[1]))?;
                Err::<(), Error>(Error::EditUnsupported("inner failed"))
            })
            .unwrap_err();
        assert!(
            matches!(err, Error::EditUnsupported("inner failed")),
            "{err:?}"
        );
        let err = outer.delete(&outer_path).unwrap_err();
        assert!(
            matches!(
                err,
                Error::EditUnsupported(
                    "an edit that changes or withdraws an already-staged object cannot be made \
                 inside an atomic staging batch, whose undo drops additions only"
                )
            ),
            "{err:?}"
        );
        if retain_outer {
            Ok(())
        } else {
            Err(Error::EditUnsupported("outer failed"))
        }
    });
    if retain_outer {
        result.unwrap();
        assert_eq!(
            engine.staged_object(&outer_path).unwrap().kind,
            StagedKind::Group
        );
        engine.delete(&outer_path).unwrap();
    } else {
        let err = result.unwrap_err();
        assert!(
            matches!(err, Error::EditUnsupported("outer failed")),
            "{err:?}"
        );
        assert!(engine.staged_object(&outer_path).is_none());
    }
    assert!(engine.staged.is_empty());
    assert!(engine.staged_object(&inner_path).is_none());
    assert!(
        engine
            .staged_object(&ObjectPathBuf::parse("inner/pending"))
            .is_none()
    );
    engine.create_group(&inner_path).unwrap();
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("inner/pending"), dataset(&[2]))
        .unwrap();
    assert_eq!(
        engine
            .staged_dataset_meta(&ObjectPathBuf::parse("inner/pending"))
            .unwrap()
            .dimensions,
        vec![1]
    );
    engine.commit().unwrap();
}

fn open_session(path: &Path) -> WriteEngine {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("appendable")
        .with_i32_data(&[1])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[1]);
    builder.create_dataset("overwritten").with_i32_data(&[1]);
    builder.create_dataset("attributed").with_i32_data(&[1]);
    builder.create_dataset("deleted").with_i32_data(&[1]);
    builder.write(path).unwrap();
    WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap()
}

fn source_file(path: &Path) -> File {
    let mut builder = FileBuilder::new();
    builder.create_dataset("source").with_i32_data(&[4]);
    builder.write(path).unwrap();
    File::open(path).unwrap()
}

impl WriteEngine {
    fn stage_operations(&mut self, source: &File) -> Result<(), Error> {
        self.stage_created_dataset(&ObjectPathBuf::parse("created"), dataset(&[7]))?;
        self.stage_dataset_write(&ObjectPathBuf::parse("overwritten"), dataset(&[2]))?;
        let mut append = AppendBuilder::new();
        append.append_i32(&[3]);
        self.stage_dataset_append(&ObjectPathBuf::parse("appendable"), append)?;
        self.create_group(&ObjectPathBuf::parse("group"))?;
        self.set_group_attr(&ObjectPathBuf::root(), "label", AttrValue::I32(5))?;
        self.set_dataset_attr(
            &ObjectPathBuf::parse("attributed"),
            "label",
            AttrValue::I32(6),
        )?;
        self.delete(&ObjectPathBuf::parse("deleted"))?;
        self.copy(
            &ObjectPathBuf::parse("overwritten"),
            &ObjectPathBuf::parse("copied"),
        )?;
        self.copy_from(
            source,
            &ObjectPathBuf::parse("source"),
            &ObjectPathBuf::parse("foreign"),
        )?;
        Ok(())
    }
}

fn dataset(values: &[i32]) -> DatasetBuilder {
    let mut builder = DatasetBuilder::new("");
    builder.with_i32_data(values);
    builder
}

fn assert_contents(path: &Path) {
    let file = File::open(path).unwrap();
    assert_eq!(
        file.dataset("created").unwrap().read_i32().unwrap(),
        vec![7]
    );
    assert_eq!(
        file.dataset("overwritten").unwrap().read_i32().unwrap(),
        vec![2]
    );
    assert_eq!(
        file.dataset("appendable").unwrap().read_i32().unwrap(),
        vec![1, 3]
    );
    assert_eq!(file.dataset("copied").unwrap().read_i32().unwrap(), vec![1]);
    assert_eq!(
        file.dataset("foreign").unwrap().read_i32().unwrap(),
        vec![4]
    );
    assert!(file.group("group").unwrap().datasets().unwrap().is_empty());
    assert_eq!(
        file.root().attrs().unwrap().get("label"),
        Some(&AttrValue::I32(5))
    );
    assert_eq!(
        file.dataset("attributed")
            .unwrap()
            .attrs()
            .unwrap()
            .get("label"),
        Some(&AttrValue::I32(6))
    );
    let err = file.dataset("deleted").unwrap_err();
    let Error::Format(FormatError::PathNotFound(missing)) = &err else {
        panic!("expected PathNotFound, got {err:?}");
    };
    assert_eq!(missing, "deleted");
}
