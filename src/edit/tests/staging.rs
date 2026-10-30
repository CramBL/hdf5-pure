//! Pending creation visibility, replacement, withdrawal, and batch rollback.

use std::path::Path;

use rstest::rstest;

use super::common;
use crate::DeflateLevel;
use crate::PersistedFreeSpacePolicy;
use crate::access_mode::AccessMode;
use crate::dataspace::MaxExtent;
use crate::edit::AppendBuilder;
use crate::edit::MemoryStrategy;
use crate::edit::WriteEngine;
use crate::edit::staging::StagedKind;
use crate::error::Error;
use crate::error::FormatError;
use crate::error::ResolveError;
use crate::file_lock::FileLocking;
use crate::layout_info::FilterId;
use crate::object_path::ObjectPath;
use crate::object_path::ObjectPathBuf;
use crate::type_builders;
use crate::type_builders::AttrValue;
use crate::type_builders::DatasetBuilder;

#[test]
fn overwriting_the_root_group_is_rejected_at_staging() {
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("root_overwrite.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1]);
    b.write(&path).unwrap();

    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let mut db = crate::type_builders::DatasetBuilder::new("whatever");
    db.with_i32_data(&[1]);
    let err = s
        .stage_dataset_write(&ObjectPathBuf::parse("/"), db)
        .unwrap_err();
    common::assert_unsupported(err, "cannot overwrite the root group");
    assert!(!s.has_staged_edits());
}

#[test]
fn staging_unallocated_storage_into_an_existing_file_preserves_the_schema() {
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unallocated.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1]);
    b.write(&path).unwrap();

    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let mut db = crate::type_builders::DatasetBuilder::new("sparse");
    db.with_unallocated_storage(type_builders::make_i32_type(), &[1000]);
    db.with_chunks(&[100]);
    s.stage_created_dataset(&ObjectPathBuf::parse("/sparse"), db)
        .unwrap();
    let meta = s
        .staged_dataset_meta(&ObjectPathBuf::parse("/sparse"))
        .unwrap();
    assert_eq!(meta.datatype, type_builders::make_i32_type());
    assert_eq!(meta.dimensions, [1000]);
    assert_eq!(meta.maxshape, None);
    assert!(meta.chunked);
    assert!(s.has_staged_edits());
}

#[test]
fn staged_queries_report_only_explicit_creations() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = open_session(&dir.path().join("q.h5"));
    e.create_group(&ObjectPathBuf::parse("a")).unwrap();
    let mut col = DatasetBuilder::new("");
    col.with_i32_data(&[1, 2])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[2]);
    e.stage_created_dataset(&ObjectPathBuf::parse("a/b/col"), col)
        .unwrap();

    assert_eq!(kind(&e, "a"), Some(StagedKind::Group));
    assert_eq!(kind(&e, "a/b/col"), Some(StagedKind::Dataset));
    assert_eq!(kind(&e, "a/b"), None);
    assert_eq!(kind(&e, ""), None);
    assert_eq!(kind(&e, "existing"), None);
    assert_eq!(kind(&e, "a/missing"), None);

    assert_eq!(
        child_names(&e, ""),
        vec![("a".to_string(), StagedKind::Group)]
    );
    assert_eq!(
        child_names(&e, "a/b"),
        vec![("col".to_string(), StagedKind::Dataset)]
    );
    assert!(child_names(&e, "a").is_empty());
    assert!(child_names(&e, "existing").is_empty());
    e.create_group(&ObjectPathBuf::parse("a/b")).unwrap();
    assert_eq!(kind(&e, "a/b"), Some(StagedKind::Group));
    assert_eq!(
        child_names(&e, "a"),
        vec![("b".to_string(), StagedKind::Group)]
    );
}

#[rstest]
#[case::shuffle(MetadataFilter::Shuffle, FilterId::Shuffle)]
#[case::deflate(MetadataFilter::Deflate, FilterId::Deflate)]
fn staged_dataset_metadata_respects_filter_availability(
    #[case] filter: MetadataFilter,
    #[case] expected: FilterId,
) {
    let path = test_util::temp::temp_path("q.h5");
    let mut e = open_session(&path);
    let mut col = DatasetBuilder::new("");
    col.with_i32_data(&[1, 2, 3, 4])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[2]);
    match filter {
        MetadataFilter::Deflate => col.with_deflate(DeflateLevel::Four),
        MetadataFilter::Shuffle => col.with_shuffle(),
    };
    let staged = e.stage_created_dataset(&ObjectPathBuf::parse("col"), col);
    if matches!(filter, MetadataFilter::Deflate)
        && !cfg!(any(feature = "deflate", feature = "fast-deflate"))
    {
        let error = staged.unwrap_err();
        let Error::EditUnsupported(reason) = error else {
            panic!("{error:?}");
        };
        assert_eq!(
            reason,
            "deflate compression requires the `deflate` crate feature"
        );
        assert!(!e.has_staged_edits());
        assert!(
            e.staged_dataset_meta(&ObjectPathBuf::parse("col"))
                .is_none()
        );
    } else {
        staged.unwrap();
        let meta = e.staged_dataset_meta(&ObjectPathBuf::parse("col")).unwrap();
        assert_eq!(meta.dimensions, vec![4]);
        assert_eq!(meta.maxshape, Some(vec![MaxExtent::Unlimited]));
        assert_eq!(meta.datatype.type_size(), 4);
        assert!(meta.chunked);
        assert_eq!(meta.filters, vec![(expected, false)]);
    }
    e.stage_created_dataset(&ObjectPathBuf::parse("plain"), i32_dataset(&[5]))
        .unwrap();
    let plain = e
        .staged_dataset_meta(&ObjectPathBuf::parse("plain"))
        .unwrap();
    assert_eq!(plain.dimensions, vec![1]);
    assert_eq!(plain.maxshape, None);
    assert!(!plain.chunked);
    assert!(plain.filters.is_empty());
    e.create_group(&ObjectPathBuf::parse("g")).unwrap();
    assert!(e.staged_dataset_meta(&ObjectPathBuf::parse("g")).is_none());
    assert!(
        e.staged_dataset_meta(&ObjectPathBuf::parse("existing"))
            .is_none()
    );
    e.commit().unwrap();
    drop(e);
    let file = crate::File::open(&path).unwrap();
    assert_eq!(file.dataset("plain").unwrap().read_i32().unwrap(), [5]);
    if !matches!(filter, MetadataFilter::Deflate)
        || cfg!(any(feature = "deflate", feature = "fast-deflate"))
    {
        assert_eq!(
            file.dataset("col").unwrap().read_i32().unwrap(),
            [1, 2, 3, 4]
        );
    }
}

#[test]
fn staged_deletion_preserves_committed_visibility_until_replacement() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = open_session(&dir.path().join("q.h5"));
    e.delete(&ObjectPathBuf::parse("existing")).unwrap();
    assert_eq!(kind(&e, "existing"), None);
    assert!(e.staged_children(&ObjectPathBuf::parse("")).is_empty());

    e.stage_created_dataset(&ObjectPathBuf::parse("existing"), i32_dataset(&[9]))
        .unwrap();
    let staged = e.staged_object(&ObjectPathBuf::parse("existing")).unwrap();
    assert_eq!(staged.kind, StagedKind::Dataset);
    assert!(staged.replaces_link, "the same commit removes the link");
    assert_eq!(
        e.staged_dataset_meta(&ObjectPathBuf::parse("existing"))
            .unwrap()
            .dimensions,
        vec![1]
    );
    assert!(e.staged_children(&ObjectPathBuf::parse(""))[0].replaces_link);
}

#[test]
fn a_creation_colliding_with_a_surviving_link_is_rejected_where_it_is_staged() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = open_session(&dir.path().join("q.h5"));
    let err = e
        .stage_created_dataset(&ObjectPathBuf::parse("existing"), i32_dataset(&[9]))
        .unwrap_err();
    common::assert_unsupported(
        err,
        "a link with this name already exists in the target group",
    );
    common::assert_unsupported(
        e.create_group(&ObjectPathBuf::parse("existing"))
            .unwrap_err(),
        "a link with this name already exists in the target group",
    );
    assert_eq!(kind(&e, "existing"), None);
    assert!(
        e.staged_dataset_meta(&ObjectPathBuf::parse("existing"))
            .is_none()
    );
    assert!(e.staged_children(&ObjectPathBuf::parse("")).is_empty());
    e.commit().unwrap();
}

#[test]
fn a_second_creation_at_a_staged_path_is_rejected_where_it_is_staged() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = open_session(&dir.path().join("q.h5"));
    e.stage_created_dataset(&ObjectPathBuf::parse("fresh"), i32_dataset(&[1, 2, 3]))
        .unwrap();
    let err = e
        .stage_created_dataset(&ObjectPathBuf::parse("fresh"), i32_dataset(&[9]))
        .unwrap_err();
    common::assert_unsupported(
        err,
        "this session already stages an object with this name in the target group; \
         delete it to withdraw that staging before staging another in its place",
    );
    assert_eq!(
        e.staged_dataset_meta(&ObjectPathBuf::parse("fresh"))
            .unwrap()
            .dimensions,
        vec![3]
    );
    assert_eq!(
        child_names(&e, ""),
        vec![("fresh".to_string(), StagedKind::Dataset)]
    );
    common::assert_unsupported(
        e.create_group(&ObjectPathBuf::parse("fresh")).unwrap_err(),
        "this session already stages an object with this name in the target group; \
         delete it to withdraw that staging before staging another in its place",
    );
    e.create_group(&ObjectPathBuf::parse("g")).unwrap();
    common::assert_unsupported(
        e.stage_created_dataset(&ObjectPathBuf::parse("g"), i32_dataset(&[9]))
            .unwrap_err(),
        "this session already stages an object with this name in the target group; \
         delete it to withdraw that staging before staging another in its place",
    );
    assert_eq!(kind(&e, "g"), Some(StagedKind::Group));

    e.commit().unwrap();
}

#[test]
fn a_replacement_is_staged_once_and_a_withdrawal_frees_the_name_again() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = open_session(&dir.path().join("q.h5"));
    e.delete(&ObjectPathBuf::parse("existing")).unwrap();
    e.stage_created_dataset(&ObjectPathBuf::parse("existing"), i32_dataset(&[9]))
        .unwrap();
    common::assert_unsupported(
        e.stage_created_dataset(&ObjectPathBuf::parse("existing"), i32_dataset(&[8, 8]))
            .unwrap_err(),
        "this session already stages an object with this name in the target group; \
         delete it to withdraw that staging before staging another in its place",
    );
    assert_eq!(
        e.staged_dataset_meta(&ObjectPathBuf::parse("existing"))
            .unwrap()
            .dimensions,
        vec![1]
    );
    e.delete(&ObjectPathBuf::parse("existing")).unwrap();
    e.stage_created_dataset(&ObjectPathBuf::parse("existing"), i32_dataset(&[8, 8]))
        .unwrap();
    assert_eq!(
        e.staged_dataset_meta(&ObjectPathBuf::parse("existing"))
            .unwrap()
            .dimensions,
        vec![2]
    );
    e.commit().unwrap();
}

#[test]
fn a_group_staged_twice_is_one_group_and_stays_allowed() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = open_session(&dir.path().join("q.h5"));
    e.create_group(&ObjectPathBuf::parse("g")).unwrap();
    e.create_group(&ObjectPathBuf::parse("g")).unwrap();
    assert_eq!(kind(&e, "g"), Some(StagedKind::Group));
    assert_eq!(
        child_names(&e, ""),
        vec![("g".to_string(), StagedKind::Group)]
    );
    e.stage_created_dataset(&ObjectPathBuf::parse("g/inner"), i32_dataset(&[7]))
        .unwrap();
    e.commit().unwrap();
}

#[test]
fn deleting_an_ancestor_alone_preserves_committed_child_visibility() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prefix.h5");
    let mut b = crate::writer::FileBuilder::new();
    let mut g = b.create_group("g");
    g.create_dataset("inner").with_i32_data(&[1]);
    b.add_group(g.finish());
    b.write(&path).unwrap();

    let mut e = WriteEngine::open_rw_with_strategy(
        &path,
        crate::source::MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Mirrored,
        PersistedFreeSpacePolicy::Tolerant,
    )
    .unwrap();
    e.delete(&ObjectPathBuf::parse("g")).unwrap();
    let err = e
        .stage_created_dataset(&ObjectPathBuf::parse("g/inner"), i32_dataset(&[9]))
        .unwrap_err();
    common::assert_unsupported(
        err,
        "a link with this name already exists in the target group",
    );
    assert_eq!(kind(&e, "g/inner"), None, "the file's own dataset");
    e.stage_created_dataset(&ObjectPathBuf::parse("g/other"), i32_dataset(&[9]))
        .unwrap();
    assert!(
        !e.staged_children(&ObjectPathBuf::parse("g"))[0].replaces_link,
        "nothing rebuilds `g`, so its names are not handed over"
    );
    common::assert_unsupported(
        e.commit().unwrap_err(),
        "a deletion overlaps an addition in the same commit; \
         replace the path instead, or use separate commits",
    );
}

#[test]
fn recreating_a_deleted_group_makes_staged_children_visible() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("replaced.h5");
    let mut b = crate::writer::FileBuilder::new();
    let mut g = b.create_group("g");
    g.create_dataset("inner").with_i32_data(&[1]);
    b.add_group(g.finish());
    b.write(&path).unwrap();

    let mut e = WriteEngine::open_rw_with_strategy(
        &path,
        crate::source::MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Mirrored,
        PersistedFreeSpacePolicy::Tolerant,
    )
    .unwrap();
    e.delete(&ObjectPathBuf::parse("g")).unwrap();
    e.create_group(&ObjectPathBuf::parse("g")).unwrap();
    e.stage_created_dataset(&ObjectPathBuf::parse("g/inner"), i32_dataset(&[9]))
        .unwrap();
    assert_eq!(kind(&e, "g/inner"), Some(StagedKind::Dataset));
    assert!(
        e.staged_object(&ObjectPathBuf::parse("g/inner"))
            .unwrap()
            .replaces_link
    );
    assert!(e.staged_children(&ObjectPathBuf::parse("g"))[0].replaces_link);
    e.commit().unwrap();
}

#[rstest]
#[case::empty("")]
#[case::absolute_root("/")]
#[case::dot(".")]
fn the_root_cannot_be_deleted(#[case] spelling: &str) {
    let dir = tempfile::tempdir().unwrap();
    let mut e = open_session(&dir.path().join("q.h5"));
    let err = e.delete(&ObjectPathBuf::parse(spelling)).unwrap_err();
    let Error::EditUnsupported(message) = &err else {
        panic!("expected EditUnsupported, got {err:?}");
    };
    assert_eq!(
        *message,
        "cannot delete the root group; delete its members instead"
    );
    e.stage_created_dataset(&ObjectPathBuf::parse("fresh"), i32_dataset(&[1]))
        .unwrap();
    assert!(
        !e.staged_object(&ObjectPathBuf::parse("fresh"))
            .unwrap()
            .replaces_link,
        "the refused deletion staged nothing, so this replaces nothing"
    );
}

#[rstest]
#[case::empty("")]
#[case::absolute_root("/")]
#[case::dot(".")]
fn the_root_group_cannot_be_created(#[case] spelling: &str) {
    let dir = tempfile::tempdir().unwrap();
    let mut e = open_session(&dir.path().join("q.h5"));
    let err = e.create_group(&ObjectPathBuf::parse(spelling)).unwrap_err();
    let Error::EditUnsupported(message) = &err else {
        panic!("expected EditUnsupported, got {err:?}");
    };
    assert_eq!(*message, "cannot create the root group");
    assert!(
        !e.has_staged_edits(),
        "the refusal staged nothing for a later commit to find"
    );
}

#[test]
fn an_append_onto_a_staged_dataset_grows_the_pending_creation() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = open_session(&dir.path().join("q.h5"));
    e.stage_created_dataset(&ObjectPathBuf::parse("col"), i32_dataset(&[1, 2]))
        .unwrap();

    let mut b = AppendBuilder::new();
    b.append_i32(&[3, 4]);
    e.stage_dataset_append_pending(&ObjectPathBuf::parse("col"), b)
        .unwrap();
    assert_eq!(
        e.staged_dataset_meta(&ObjectPathBuf::parse("col"))
            .unwrap()
            .dimensions,
        vec![4]
    );
    assert!(e.staged.appends().is_empty());

    e.commit().unwrap();
    let addr = crate::group_v2::resolve_path_any_from_source(
        &e.image(),
        AccessMode::ReadWrite,
        e.superblock(),
        e.widths(),
        e.addresses(),
        &ObjectPath::parse("col"),
    )
    .unwrap();
    assert!(addr > 0);
}

#[test]
fn an_append_through_a_handle_onto_the_replaced_object_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = open_session(&dir.path().join("q.h5"));
    e.delete(&ObjectPathBuf::parse("existing")).unwrap();
    e.stage_created_dataset(&ObjectPathBuf::parse("existing"), i32_dataset(&[100]))
        .unwrap();
    let mut b = AppendBuilder::new();
    b.append_i32(&[200]);
    common::assert_unsupported(
        e.stage_dataset_append(&ObjectPathBuf::parse("existing"), b)
            .unwrap_err(),
        "a dataset is staged at this path in the same commit, so an append through a \
         handle onto the object the file holds there could only grow the object being \
         replaced; append through the handle that staged the creation, or commit first",
    );
    assert_eq!(
        e.staged_dataset_meta(&ObjectPathBuf::parse("existing"))
            .unwrap()
            .dimensions,
        vec![1],
        "the refusal must leave the replacement alone"
    );
    assert!(e.staged.appends().is_empty());
}

#[test]
fn incompatible_staged_appends_preserve_pending_dataset_shape() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = open_session(&dir.path().join("q.h5"));
    e.stage_created_dataset(&ObjectPathBuf::parse("col"), i32_dataset(&[1, 2]))
        .unwrap();

    let mut wrong_type = AppendBuilder::new();
    wrong_type.append_f64(&[1.0]);
    let error = e
        .stage_dataset_append_pending(&ObjectPathBuf::parse("col"), wrong_type)
        .unwrap_err();
    let Error::AppendUnsupported(reason) = error else {
        panic!("expected AppendUnsupported, got {error:?}");
    };
    assert_eq!(
        reason,
        "appended element datatype does not match the staged dataset's datatype"
    );

    let mut partial = AppendBuilder::new();
    partial.append_raw(&[0u8, 1, 2]);
    let error = e
        .stage_dataset_append_pending(&ObjectPathBuf::parse("col"), partial)
        .unwrap_err();
    let Error::AppendUnsupported(reason) = error else {
        panic!("expected AppendUnsupported, got {error:?}");
    };
    assert_eq!(
        reason,
        "appended byte length is not a whole number of rows of the staged dataset"
    );
    let mut capped = DatasetBuilder::new("");
    capped
        .with_i32_data(&[1, 2])
        .with_maxshape(&[MaxExtent::Fixed(3)])
        .with_chunks(&[2]);
    e.stage_created_dataset(&ObjectPathBuf::parse("capped"), capped)
        .unwrap();
    let mut over = AppendBuilder::new();
    over.append_i32(&[3, 4]);
    let error = e
        .stage_dataset_append_pending(&ObjectPathBuf::parse("capped"), over)
        .unwrap_err();
    let Error::AppendUnsupported(reason) = error else {
        panic!("expected AppendUnsupported, got {error:?}");
    };
    assert_eq!(
        reason,
        "appending would grow the staged dataset past its maximum shape"
    );
    assert_eq!(
        e.staged_dataset_meta(&ObjectPathBuf::parse("col"))
            .unwrap()
            .dimensions,
        vec![2]
    );
    assert_eq!(
        e.staged_dataset_meta(&ObjectPathBuf::parse("capped"))
            .unwrap()
            .dimensions,
        vec![2]
    );
}

#[test]
fn committed_dataset_appends_remain_separate_staged_operations() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = open_session(&dir.path().join("q.h5"));
    let mut b = AppendBuilder::new();
    b.append_i32(&[4]);
    e.stage_dataset_append(&ObjectPathBuf::parse("existing"), b)
        .unwrap();
    assert_eq!(e.staged.appends().len(), 1);
    let mut b = AppendBuilder::new();
    b.append_i32(&[5]);
    e.stage_dataset_append_pending(&ObjectPathBuf::parse("existing"), b)
        .unwrap();
    assert_eq!(e.staged.appends().len(), 2);
}

#[test]
fn deleting_a_staged_creation_withdraws_it() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = open_session(&dir.path().join("q.h5"));
    e.stage_created_dataset(&ObjectPathBuf::parse("col"), i32_dataset(&[1, 2]))
        .unwrap();
    e.delete(&ObjectPathBuf::parse("col")).unwrap();
    assert_eq!(kind(&e, "col"), None);
    assert!(e.staged.deletes().is_empty());
    assert!(!e.has_staged_edits());
    e.commit().unwrap();
}

#[test]
fn deleting_a_staged_group_withdraws_its_staged_subtree() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = open_session(&dir.path().join("q.h5"));
    e.create_group(&ObjectPathBuf::parse("g")).unwrap();
    e.create_group(&ObjectPathBuf::parse("g/inner")).unwrap();
    e.stage_created_dataset(&ObjectPathBuf::parse("g/inner/col"), i32_dataset(&[1]))
        .unwrap();
    e.set_group_attr(&ObjectPathBuf::parse("g"), "kind", AttrValue::I64(1))
        .unwrap();

    e.delete(&ObjectPathBuf::parse("g")).unwrap();
    assert_eq!(kind(&e, "g"), None);
    assert_eq!(kind(&e, "g/inner"), None);
    assert_eq!(kind(&e, "g/inner/col"), None);
    assert!(
        !e.has_staged_edits(),
        "the attribute and the subtree go with the group"
    );
    e.commit().unwrap();
}

#[test]
fn deleting_a_staged_replacement_leaves_the_plain_deletion() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = open_session(&dir.path().join("q.h5"));
    e.delete(&ObjectPathBuf::parse("existing")).unwrap();
    e.stage_created_dataset(&ObjectPathBuf::parse("existing"), i32_dataset(&[9]))
        .unwrap();
    e.delete(&ObjectPathBuf::parse("existing")).unwrap();
    assert_eq!(kind(&e, "existing"), None);
    assert_eq!(e.staged.deletes().len(), 1);
    e.commit().unwrap();
    let error = crate::group_v2::resolve_path_any_from_source(
        &e.image(),
        AccessMode::ReadWrite,
        e.superblock(),
        e.widths(),
        e.addresses(),
        &ObjectPath::parse("existing"),
    )
    .unwrap_err();
    let ResolveError::Format(FormatError::PathNotFound(path)) = error else {
        panic!("expected PathNotFound, got {error:?}");
    };
    assert_eq!(path, "existing");
}

#[test]
fn a_batch_that_fails_leaves_the_staged_index_matching_the_staged_set() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = open_session(&dir.path().join("q.h5"));
    e.stage_created_dataset(&ObjectPathBuf::parse("kept"), i32_dataset(&[1]))
        .unwrap();
    let refused: Result<(), Error> = e.stage_atomically(|s| {
        s.create_group(&ObjectPathBuf::parse("gone"))?;
        s.stage_created_dataset(&ObjectPathBuf::parse("gone/col"), i32_dataset(&[2]))?;
        Err(Error::EditUnsupported("refused on purpose"))
    });
    common::assert_unsupported(refused.unwrap_err(), "refused on purpose");
    assert_eq!(kind(&e, "gone"), None);
    assert_eq!(kind(&e, "gone/col"), None);
    assert_eq!(kind(&e, "kept"), Some(StagedKind::Dataset));
    assert!(e.staged_children(&ObjectPathBuf::parse("gone")).is_empty());
    e.create_group(&ObjectPathBuf::parse("other")).unwrap();
    e.stage_created_dataset(&ObjectPathBuf::parse("other/col"), i32_dataset(&[3]))
        .unwrap();
    e.create_group(&ObjectPathBuf::parse("gone")).unwrap();
    e.stage_created_dataset(&ObjectPathBuf::parse("gone/col"), i32_dataset(&[2]))
        .unwrap();
    assert_eq!(kind(&e, "gone"), Some(StagedKind::Group));
    assert_eq!(kind(&e, "gone/col"), Some(StagedKind::Dataset));
    assert_eq!(
        e.staged_dataset_meta(&ObjectPathBuf::parse("gone/col"))
            .unwrap()
            .dimensions,
        vec![1]
    );
    e.commit().unwrap();
}

#[test]
fn an_edit_that_changes_staged_work_is_rejected_inside_a_batch() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = open_session(&dir.path().join("q.h5"));
    e.stage_created_dataset(&ObjectPathBuf::parse("col"), i32_dataset(&[1, 2]))
        .unwrap();
    let folded: Result<(), Error> = e.stage_atomically(|s| {
        let mut b = AppendBuilder::new();
        b.append_i32(&[3]);
        s.stage_dataset_append_pending(&ObjectPathBuf::parse("col"), b)
    });
    common::assert_unsupported(
        folded.unwrap_err(),
        "an edit that changes or withdraws an already-staged object cannot be made \
         inside an atomic staging batch, whose undo drops additions only",
    );
    let withdrawn: Result<(), Error> =
        e.stage_atomically(|s| s.delete(&ObjectPathBuf::parse("col")));
    common::assert_unsupported(
        withdrawn.unwrap_err(),
        "an edit that changes or withdraws an already-staged object cannot be made \
         inside an atomic staging batch, whose undo drops additions only",
    );
    assert_eq!(
        e.staged_dataset_meta(&ObjectPathBuf::parse("col"))
            .unwrap()
            .dimensions,
        vec![2]
    );
}

/// Writes dataset `existing` and opens a mirrored editor with its metadata cache disabled.
fn open_session(path: &Path) -> WriteEngine {
    let mut b = crate::writer::FileBuilder::new();
    b.create_dataset("existing").with_i32_data(&[1, 2, 3]);
    b.write(path).unwrap();
    WriteEngine::open_rw_with_strategy(
        path,
        crate::source::MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Mirrored,
        PersistedFreeSpacePolicy::Tolerant,
    )
    .unwrap()
}

fn i32_dataset(data: &[i32]) -> DatasetBuilder {
    let mut b = DatasetBuilder::new("");
    b.with_i32_data(data);
    b
}

fn kind(e: &WriteEngine, path: &str) -> Option<StagedKind> {
    e.staged_object(&ObjectPathBuf::parse(path)).map(|o| o.kind)
}

fn child_names(e: &WriteEngine, parent: &str) -> Vec<(String, StagedKind)> {
    e.staged_children(&ObjectPathBuf::parse(parent))
        .into_iter()
        .map(|c| (c.name, c.kind))
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum MetadataFilter {
    Deflate,
    Shuffle,
}
