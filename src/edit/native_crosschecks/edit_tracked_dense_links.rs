//! Tracked group-link crosschecks against libhdf5.

use std::path::Path;

use hdf5_pure_format::__private::BTREE_V2_LINK_CREATION_ORDER;
use hdf5_pure_format::__private::BTreeV2Header;
use hdf5_pure_format::__private::GroupLinkStorage;
use hdf5_pure_format::__private::LinkInfoMessage;
use hdf5_pure_format::__private::ObjectHeaderPrefix;
use hdf5_pure_format::__private::OffsetWidth;
use hdf5_pure_format::__private::OwnedLinkTarget;
use rstest::rstest;
use test_util_hdf5::creation_order;
use test_util_hdf5::creation_order::Indexing;
use test_util_hdf5::dense_links::GroupOrder;
use test_util_hdf5::file;

use crate::AttrValue;
use crate::CharacterSet;
use crate::Error;
use crate::File;
use crate::FileBuilder;
use crate::MessageType;
use crate::Object;
use crate::edit::group_inspection;

#[rstest]
#[case::compact_boundary(8, 0, 1, (8, 6), true)]
#[case::dense_boundary(9, 3, 0, (8, 6), true)]
#[case::dense_to_compact(9, 4, 0, (8, 6), false)]
#[case::staged_final_count(9, 4, 4, (8, 6), true)]
#[case::custom_compact_boundary(4, 0, 1, (4, 3), true)]
#[case::custom_dense_boundary(5, 2, 0, (4, 3), true)]
#[case::custom_dense_to_compact(5, 3, 0, (4, 3), false)]
#[case::empty_dense(9, 9, 0, (8, 0), true)]
#[case::empty_compact(9, 9, 0, (8, 6), false)]
fn tracked_storage_transitions_preserve_indexes_and_the_next_counter(
    #[case] count: usize,
    #[case] removed: usize,
    #[case] added: usize,
    #[case] thresholds: (u32, u32),
    #[case] dense: bool,
    #[values(Indexing::No, Indexing::Yes)] indexing: Indexing,
    #[values("/", "/parent/g")] group_path: &str,
    #[values(false, true)] source_only: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("transition.h5");
    creation_order::write_link_fixture(&path, group_path, count, indexing, thresholds);
    let file = group_inspection::open_group_inspection(&path, source_only).unwrap();
    let group = file.group(group_path).unwrap();
    for index in 0..removed {
        group.delete(&format!("link_{index:04}")).unwrap();
    }
    for index in 0..added {
        group.create_group(&format!("added_{index:04}")).unwrap();
    }
    file.commit().unwrap();
    drop(group);
    drop(file);
    let expected = (removed..count)
        .map(|index| (format!("link_{index:04}"), u64::try_from(index).unwrap()))
        .chain((0..added).map(|index| {
            (
                format!("added_{index:04}"),
                u64::try_from(count + index).unwrap(),
            )
        }))
        .collect::<Vec<_>>();
    assert_group(
        &path,
        group_path,
        &expected,
        u64::try_from(count + added).unwrap(),
        indexing,
        dense,
        source_only,
    );
    creation_order::assert_link_dataset_values(&path, group_path, removed, count);
    dir.close().unwrap();
}

#[rstest]
fn tracked_creation_indexes_keep_deletion_gaps_across_repeated_transitions(
    #[values(Indexing::No, Indexing::Yes)] indexing: Indexing,
    #[values(false, true)] source_only: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gaps.h5");
    creation_order::write_link_fixture(&path, "/", 9, indexing, (8, 6));
    let file = group_inspection::open_group_inspection(&path, source_only).unwrap();
    for index in 1..6 {
        file.root().delete(&format!("link_{index:04}")).unwrap();
    }
    file.commit().unwrap();
    drop(file);
    let mut expected = [0, 6, 7, 8]
        .into_iter()
        .map(|index| (format!("link_{index:04}"), index))
        .collect::<Vec<_>>();
    assert_group(&path, "/", &expected, 9, indexing, false, source_only);
    let file = group_inspection::open_group_inspection(&path, source_only).unwrap();
    file.root().delete("link_0008").unwrap();
    file.commit().unwrap();
    for index in 0..5 {
        file.root()
            .create_group(&format!("added_{index:04}"))
            .unwrap();
    }
    file.commit().unwrap();
    drop(file);
    expected.pop();
    expected.extend((0..5).map(|index| (format!("added_{index:04}"), 9 + index)));
    assert_group(&path, "/", &expected, 14, indexing, false, source_only);
    let file = group_inspection::open_group_inspection(&path, source_only).unwrap();
    file.root().create_group("last").unwrap();
    file.commit().unwrap();
    drop(file);
    expected.push(("last".into(), 14));
    assert_group(&path, "/", &expected, 15, indexing, true, source_only);
    creation_order::add_reference_group(&path, "from_libhdf5");
    expected.push(("from_libhdf5".into(), 15));
    assert_group(&path, "/", &expected, 16, indexing, true, source_only);
    dir.close().unwrap();
}

#[rstest]
fn tracked_group_additions_and_replacements_assign_monotonic_indexes(
    #[values(4, 9)] count: usize,
    #[values(Indexing::No, Indexing::Yes)] indexing: Indexing,
    #[values(false, true)] source_only: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("additions.h5");
    creation_order::write_link_fixture(&path, "/", count, indexing, (8, 6));
    let source_path = dir.path().join("source.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("source").with_i32_data(&[17, 23]);
    builder.write(&source_path).unwrap();
    let source = File::open(&source_path).unwrap();
    let file = group_inspection::open_group_inspection(&path, source_only).unwrap();
    file.root().delete("link_0000").unwrap();
    file.root()
        .create_dataset("link_0000", |dataset| {
            dataset.with_i32_data(&[99]);
        })
        .unwrap();
    file.copy_from(&source, "source", "imported").unwrap();
    file.root()
        .create_dataset("refs", |dataset| {
            dataset.with_path_references(&["dataset"]);
        })
        .unwrap();
    file.copy("link_0001", "copy").unwrap();
    file.root().create_group("added").unwrap();
    file.root()
        .create_dataset("dataset", |dataset| {
            dataset.with_i32_data(&[11]);
        })
        .unwrap();
    file.commit().unwrap();
    drop(file);
    drop(source);
    let next = u64::try_from(count).unwrap();
    let mut expected = (1..next)
        .map(|index| (format!("link_{index:04}"), index))
        .collect::<Vec<_>>();
    expected.extend([
        ("copy".into(), next),
        ("imported".into(), next + 1),
        ("link_0000".into(), next + 2),
        ("dataset".into(), next + 3),
        ("refs".into(), next + 4),
        ("added".into(), next + 5),
    ]);
    assert_group(&path, "/", &expected, next + 6, indexing, true, source_only);
    let reference = file::libhdf5_open(&path);
    assert_eq!(
        reference
            .dataset("imported")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [17, 23]
    );
    assert_eq!(
        reference
            .dataset("dataset")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [11]
    );
    reference.close().unwrap();
    let file = File::open(&path).unwrap();
    let targets = file.dataset("refs").unwrap().dereference().unwrap();
    let [Object::Dataset(target)] = targets.as_slice() else {
        panic!("expected a dataset reference");
    };
    assert_eq!(target.read_i32().unwrap(), [11]);
    drop(targets);
    drop(file);
    creation_order::assert_link_replacements(&path);
    dir.close().unwrap();
}

#[rstest]
fn tracked_dense_ancestors_preserve_indexes_when_a_dataset_header_moves(
    #[values(Indexing::No, Indexing::Yes)] indexing: Indexing,
    #[values(false, true)] source_only: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ancestors.h5");
    creation_order::write_link_ancestor_fixture(&path, indexing);
    let before_root = group_inspection::inspect_group_links(&path, "/", source_only).unwrap();
    let before_child = group_inspection::inspect_group_links(&path, "/child", source_only).unwrap();
    let file = group_inspection::open_group_inspection(&path, source_only).unwrap();
    file.dataset("child/link_0003")
        .unwrap()
        .set_attr("added", AttrValue::I64(99))
        .unwrap();
    file.commit().unwrap();
    drop(file);
    let root = group_inspection::inspect_group_links(&path, "/", source_only).unwrap();
    let child = group_inspection::inspect_group_links(&path, "/child", source_only).unwrap();
    assert_ne!(
        root.object_header_address,
        before_root.object_header_address
    );
    assert_ne!(
        child.object_header_address,
        before_child.object_header_address
    );
    assert_eq!(indexes(&root.links), indexes(&before_root.links));
    assert_eq!(indexes(&child.links), indexes(&before_child.links));
    creation_order::assert_link_ancestor_dataset(&path);
    dir.close().unwrap();
}

#[rstest]
fn tracked_multilevel_indexes_preserve_symbolic_links_and_creation_order(
    #[values(Indexing::No, Indexing::Yes)] indexing: Indexing,
    #[values(false, true)] source_only: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multilevel.h5");
    let order = if indexing == Indexing::Yes {
        GroupOrder::Indexed
    } else {
        GroupOrder::Tracked
    };
    test_util_hdf5::dense_links::write_inspection_fixture(&path, order);
    let group_path = format!("dense_{}", order.label());
    let before = group_inspection::inspect_group_links(&path, &group_path, source_only).unwrap();
    let mut expected = before
        .links
        .links()
        .iter()
        .filter(|link| link.name() != b"hard")
        .cloned()
        .collect::<Vec<_>>();
    let next = before.links.info().max_creation_order.unwrap();
    let file = group_inspection::open_group_inspection(&path, source_only).unwrap();
    let group = file.group(&group_path).unwrap();
    group.delete("hard").unwrap();
    group.create_group("added").unwrap();
    file.commit().unwrap();
    drop(group);
    drop(file);
    let after = group_inspection::inspect_group_links(&path, &group_path, source_only).unwrap();
    let added = after
        .links
        .links()
        .iter()
        .find(|link| link.name() == b"added")
        .unwrap();
    assert_eq!(added.creation_order(), Some(next));
    let mut actual = after
        .links
        .links()
        .iter()
        .filter(|link| link.name() != b"added")
        .cloned()
        .collect::<Vec<_>>();
    expected.sort_by(|a, b| a.name().cmp(b.name()));
    actual.sort_by(|a, b| a.name().cmp(b.name()));
    assert_eq!(actual, expected);
    let soft = after
        .links
        .links()
        .iter()
        .find(|link| link.name() == b"soft")
        .unwrap();
    assert_eq!(soft.target(), &OwnedLinkTarget::soft(b"/target".to_vec()));
    let external = after
        .links
        .links()
        .iter()
        .find(|link| link.name() == b"external")
        .unwrap();
    assert_eq!(
        external.target(),
        &OwnedLinkTarget::external("外部.h5".as_bytes().to_vec(), "/目标".as_bytes().to_vec())
    );
    let unicode = after
        .links
        .links()
        .iter()
        .find(|link| link.name() == "utf8_λ".as_bytes())
        .unwrap();
    assert_eq!(unicode.charset(), CharacterSet::Utf8);
    let mut ordered = indexes(&after.links);
    ordered.sort_by_key(|link| link.1);
    assert_group(
        &path,
        &group_path,
        &ordered,
        next + 1,
        indexing,
        true,
        source_only,
    );
    let bytes = std::fs::read(&path).unwrap();
    let info = after.links.info();
    let name_index = BTreeV2Header::parse(
        &bytes,
        512 + usize::try_from(info.btree_name_index_address.unwrap().get()).unwrap(),
        8,
        8,
    )
    .unwrap();
    assert!(name_index.depth >= 2, "{name_index:?}");
    if let Some(Some(address)) = info.btree_creation_order_address {
        let index =
            BTreeV2Header::parse(&bytes, 512 + usize::try_from(address.get()).unwrap(), 8, 8)
                .unwrap();
        assert_eq!(index.tree_type, BTREE_V2_LINK_CREATION_ORDER);
        assert!(index.depth > 0, "{index:?}");
        assert_eq!(index.total_records, 2000);
    }
    dir.close().unwrap();
}

#[rstest]
fn exhausted_dense_creation_counters_reject_the_batch_before_writing(
    #[values(Indexing::No, Indexing::Yes)] indexing: Indexing,
    #[values(false, true)] source_only: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("exhausted.h5");
    creation_order::write_link_fixture(&path, "/", 12, indexing, (8, 6));
    let inspection = group_inspection::inspect_group_links(&path, "/", source_only).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    let root = 512 + usize::try_from(inspection.object_header_address.get()).unwrap();
    let prefix = ObjectHeaderPrefix::parse(&bytes[root..]).unwrap();
    let end = root + prefix.len + usize::try_from(prefix.chunk0_size).unwrap();
    let mut pos = root + prefix.len;
    let mut replaced = false;
    while let Some(record) = prefix
        .prefix
        .layout
        .next_message(&bytes[..end], pos)
        .unwrap()
    {
        if record.msg_type == MessageType::LINK_INFO {
            let mut info =
                LinkInfoMessage::parse_with_offset_width(record.body, OffsetWidth::Eight).unwrap();
            info.max_creation_order = Some(u64::try_from(i64::MAX).unwrap());
            let serialized = info.serialize(OffsetWidth::Eight);
            let range = record.body_range;
            bytes[range].copy_from_slice(&serialized);
            replaced = true;
            break;
        }
        pos = record.body_range.end;
    }
    assert!(replaced);
    test_util::checksum::restamp(&mut bytes, root, end + test_util::checksum::CHECKSUM - root);
    std::fs::write(&path, &bytes).unwrap();
    let file = group_inspection::open_group_inspection(&path, source_only).unwrap();
    file.root().delete("link_0000").unwrap();
    file.root().create_group("added").unwrap();
    let error = file.commit().unwrap_err();
    assert!(
        matches!(
            error,
            Error::EditUnsupported(
                "a group has assigned every link creation index its link-info message can record"
            )
        ),
        "{error:?}"
    );
    assert_eq!(file.group_inspection_snapshot().unwrap(), bytes);
    assert_eq!(file.page_publication_snapshot().unwrap(), bytes);
    drop(file);
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    dir.close().unwrap();
}

fn indexes(links: &GroupLinkStorage) -> Vec<(String, u64)> {
    let mut indexes = links
        .links()
        .iter()
        .map(|link| {
            (
                String::from_utf8(link.name().to_vec()).unwrap(),
                link.creation_order().unwrap(),
            )
        })
        .collect::<Vec<_>>();
    indexes.sort_by_key(|link| link.1);
    indexes
}

fn assert_group(
    path: &Path,
    group_path: &str,
    expected: &[(String, u64)],
    next: u64,
    indexing: Indexing,
    dense: bool,
    source_only: bool,
) {
    let inspection = group_inspection::inspect_group_links(path, group_path, source_only).unwrap();
    assert_eq!(indexes(&inspection.links), expected);
    let info = inspection.links.info();
    assert_eq!(info.max_creation_order, Some(next));
    assert_eq!(info.fractal_heap_address.is_some(), dense);
    assert_eq!(info.btree_name_index_address.is_some(), dense);
    assert_eq!(
        info.btree_creation_order_address
            .map(|address| address.is_some()),
        (indexing == Indexing::Yes).then_some(dense)
    );
    creation_order::assert_link_group(path, group_path, expected, next, indexing, dense);
}
