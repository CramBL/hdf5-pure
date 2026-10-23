//! Storage mutation tests with access to the editor's private analyses.

use std::collections::BTreeSet;

use test_util::temp;

use crate::edit::{GlobalHeapLiveness, SpaceOwnership};
use crate::error::Error;
use crate::reader::{Backend, File};
use crate::{AttrValue, FileBuilder};

mod classic_aggregation;
mod file_space;
mod paged_write;
mod persisted_free_space;
mod reclamation;

#[test]
fn deleting_a_subtree_removes_its_three_reachable_collections() {
    let path = temp::temp_path("subtree_owned_collections.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    file.root()
        .create_group_with("g", |group| {
            group.set_attr(
                "labels",
                AttrValue::VarLenStringArray(vec!["alpha".into(), "beta".into()]),
            );
            group.create_dataset("x", |builder| {
                builder.with_vlen_strings(&["alpha", "beta"]);
            });
            group.create_group_with("sub", |nested| {
                nested.create_dataset("y", |builder| {
                    builder.with_vlen_strings(&["alpha", "beta"]);
                });
            });
        })
        .unwrap();
    file.commit().unwrap();
    let before = liveness(&file).unwrap();
    assert!(before.analysis_complete);
    assert_eq!(before.referenced_collections.len(), 3);
    file.root().delete("g").unwrap();
    file.commit().unwrap();
    let after = liveness(&file).unwrap();
    assert!(after.analysis_complete);
    assert_eq!(after.referenced_collections, BTreeSet::new());
}

/// Analyzes committed storage through the file's editing session.
///
/// # Errors
///
/// Returns [`Error::ReadOnly`] if the file has no editing session, or the error from
/// [`WriteEngine::space_ownership`](crate::edit::WriteEngine::space_ownership).
///
/// # Panics
///
/// Panics if the editing-session mutex is poisoned.
fn ownership(file: &File) -> Result<SpaceOwnership, Error> {
    let Backend::Edit(engine) = &file.inner.backend else {
        return Err(Error::ReadOnly);
    };
    engine.lock().unwrap().space_ownership()
}

/// Analyzes global heap references through the file's editing session.
///
/// # Errors
///
/// Returns [`Error::ReadOnly`] if the file has no editing session, or the error from
/// [`WriteEngine::global_heap_liveness`](crate::edit::WriteEngine::global_heap_liveness).
///
/// # Panics
///
/// Panics if the editing-session mutex is poisoned.
fn liveness(file: &File) -> Result<GlobalHeapLiveness, Error> {
    let Backend::Edit(engine) = &file.inner.backend else {
        return Err(Error::ReadOnly);
    };
    engine.lock().unwrap().global_heap_liveness()
}
