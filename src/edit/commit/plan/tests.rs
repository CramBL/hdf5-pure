//! Commit preparation and staged-input ownership across application and rejection.

use rstest::rstest;

use super::{CommitPreparation, PreparedCommit};
use crate::File;
use crate::edit::WriteEngine;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::object_path::ObjectPathBuf;
use crate::type_builders::DatasetBuilder;
use crate::writer::FileBuilder;

#[rstest]
#[case::in_place(false)]
#[case::tree(true)]
fn prepared_applications_retain_their_own_staged_inputs(#[case] create: bool) {
    let directory = tempfile::tempdir().unwrap();
    let first_path = directory.path().join("first.h5");
    let second_path = directory.path().join("second.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("d").with_i32_data(&[1, 2]);
    builder.write(&first_path).unwrap();
    std::fs::copy(&first_path, &second_path).unwrap();
    let mut first = edit_test_util::open_engine(&first_path);
    let mut second = edit_test_util::open_engine(&second_path);
    first
        .stage_dataset_write(&ObjectPathBuf::parse("d"), i32_dataset(&[3, 4]))
        .unwrap();
    second
        .stage_dataset_write(&ObjectPathBuf::parse("d"), i32_dataset(&[5, 6]))
        .unwrap();
    if create {
        first
            .stage_created_dataset(&ObjectPathBuf::parse("first"), i32_dataset(&[7]))
            .unwrap();
        second
            .stage_created_dataset(&ObjectPathBuf::parse("second"), i32_dataset(&[8, 9]))
            .unwrap();
    }
    let first_staged = std::mem::take(&mut first.staged);
    let second_staged = std::mem::take(&mut second.staged);
    first.begin_commit(&first_staged);
    second.begin_commit(&second_staged);
    let CommitPreparation::Prepared(first_prepared) =
        PreparedCommit::prepare(&mut first, first_staged)
    else {
        panic!("expected the first staged set to prepare");
    };
    let CommitPreparation::Prepared(second_prepared) =
        PreparedCommit::prepare(&mut second, second_staged)
    else {
        panic!("expected the second staged set to prepare");
    };
    second.apply_commit(second_prepared).unwrap();
    first.apply_commit(first_prepared).unwrap();
    drop(second);
    drop(first);
    let first = File::open(&first_path).unwrap();
    let second = File::open(&second_path).unwrap();
    assert_eq!(first.dataset("d").unwrap().read_i32().unwrap(), [3, 4]);
    assert_eq!(second.dataset("d").unwrap().read_i32().unwrap(), [5, 6]);
    if create {
        assert_eq!(first.dataset("first").unwrap().read_i32().unwrap(), [7]);
        assert_eq!(
            second.dataset("second").unwrap().read_i32().unwrap(),
            [8, 9]
        );
        assert_eq!(first.root().datasets().unwrap(), ["d", "first"]);
        assert_eq!(second.root().datasets().unwrap(), ["d", "second"]);
    }
}

#[rstest]
#[case::fixed(Replacement::Fixed)]
#[case::variable_length(Replacement::VariableLength)]
fn rejected_preparation_retains_replacement_buffers_for_retry(#[case] replacement: Replacement) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("rejected.h5");
    let mut builder = FileBuilder::new();
    match replacement {
        Replacement::Fixed => builder.create_dataset("d").with_i32_data(&[1, 2]),
        Replacement::VariableLength => builder
            .create_dataset("d")
            .with_vlen_strings(&["before", "values"]),
    };
    builder.create_dataset("bad").with_i32_data(&[1]);
    builder.write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    let mut values = DatasetBuilder::new("");
    match replacement {
        Replacement::Fixed => values.with_i32_data(&[3, 4]),
        Replacement::VariableLength => values.with_vlen_strings(&["replacement", "strings"]),
    };
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("d"), values)
        .unwrap();
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("bad"), i32_dataset(&[5, 6]))
        .unwrap();
    let snapshot = |engine: &WriteEngine| {
        engine
            .staged
            .writes()
            .iter()
            .map(|(path, values)| {
                (
                    path.clone(),
                    values.raw().as_ptr(),
                    values.raw().to_vec(),
                    values.vl_string_staging().map(|staging| {
                        (
                            staging.patch_offsets.clone(),
                            staging
                                .collections
                                .iter()
                                .map(|collection| (collection.as_ptr(), collection.clone()))
                                .collect::<Vec<_>>(),
                        )
                    }),
                )
            })
            .collect::<Vec<_>>()
    };
    let expected = snapshot(&engine);
    let error = engine.commit().unwrap_err();
    let Error::EditUnsupported(reason) = error else {
        panic!("expected an overwrite shape error, got {error:?}");
    };
    assert_eq!(
        reason,
        "write_dataset shape does not match the on-disk dataset (overwrite, not reshape)"
    );
    assert_eq!(snapshot(&engine), expected);
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let error = engine.commit().unwrap_err();
    let Error::EditUnsupported(reason) = error else {
        panic!("expected the same overwrite shape error on retry, got {error:?}");
    };
    assert_eq!(
        reason,
        "write_dataset shape does not match the on-disk dataset (overwrite, not reshape)"
    );
    assert_eq!(snapshot(&engine), expected);
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

fn i32_dataset(values: &[i32]) -> DatasetBuilder {
    let mut builder = DatasetBuilder::new("");
    builder.with_i32_data(values);
    builder
}

#[derive(Clone, Copy)]
enum Replacement {
    Fixed,
    VariableLength,
}
