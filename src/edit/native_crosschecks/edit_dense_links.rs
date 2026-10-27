use hdf5_pure_format::__private::BTreeV2Header;
use hdf5_pure_format::__private::FractalHeapHeader;
use hdf5_pure_format::__private::GroupLinkStorage;
use hdf5_pure_format::__private::OwnedLinkTarget;
use rstest::rstest;
use test_util_hdf5::dense_links;

use crate::CharacterSet;
use crate::edit::group_inspection;

#[rstest]
#[case::compact_boundary(8, 1, 0, None, true)]
#[case::compact_retained(8, 1, 1, None, false)]
#[case::dense_boundary(9, 0, 3, None, true)]
#[case::dense_to_compact(9, 0, 4, None, false)]
#[case::custom_compact_boundary(12, 1, 0, Some((12, 4)), true)]
#[case::custom_dense_boundary(13, 0, 9, Some((12, 4)), true)]
#[case::custom_dense_to_compact(13, 0, 10, Some((12, 4)), false)]
#[case::empty_compact(9, 0, 9, None, false)]
#[case::empty_dense(1, 0, 1, Some((0, 0)), true)]
fn dense_link_edits_select_storage_from_the_final_count(
    #[case] count: usize,
    #[case] added: usize,
    #[case] removed: usize,
    #[case] thresholds: Option<(u32, u32)>,
    #[case] dense: bool,
    #[values("/", "/outer/g")] group_path: &str,
    #[values(false, true)] source_only: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("links.h5");
    dense_links::write_edit_fixture(&path, group_path, count, thresholds);
    let file = group_inspection::open_group_inspection(&path, source_only).unwrap();
    let group = file.group(group_path).unwrap();
    for index in 0..removed {
        group.delete(&format!("link_{index:04}")).unwrap();
    }
    for index in 0..added {
        group.create_group(&format!("added_{index}")).unwrap();
    }
    file.commit().unwrap();
    drop(group);
    drop(file);
    dense_links::assert_edit_group(&path, group_path, count, added, removed, dense);
    dir.close().unwrap();
}

#[rstest]
#[case::root("/")]
#[case::nested("/outer/g")]
fn dense_link_mutation_preserves_multilevel_indexes_collisions_and_symbolic_targets(
    #[case] group_path: &str,
    #[values(false, true)] source_only: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large.h5");
    dense_links::write_collision_edit_fixture(&path, group_path);
    let before = group_inspection::inspect_group_links(&path, group_path, source_only).unwrap();
    let file = group_inspection::open_group_inspection(&path, source_only).unwrap();
    let group = file.group(group_path).unwrap();
    group.delete("link_0000").unwrap();
    group.create_group("added_λ").unwrap();
    group
        .set_attr("changed", crate::AttrValue::I64(17))
        .unwrap();
    file.commit().unwrap();
    drop(group);
    drop(file);
    let after = group_inspection::inspect_group_links(&path, group_path, source_only).unwrap();
    let mut expected = before
        .links
        .links()
        .iter()
        .filter(|link| link.name() != b"link_0000")
        .cloned()
        .collect::<Vec<_>>();
    let mut actual = after
        .links
        .links()
        .iter()
        .filter(|link| link.name() != "added_λ".as_bytes())
        .cloned()
        .collect::<Vec<_>>();
    expected.sort_by(|a, b| a.name().cmp(b.name()));
    actual.sort_by(|a, b| a.name().cmp(b.name()));
    assert_eq!(actual, expected);
    let bytes = std::fs::read(&path).unwrap();
    let info = after.links.info();
    let index = BTreeV2Header::parse(
        &bytes,
        usize::try_from(info.btree_name_index_address.unwrap().get()).unwrap(),
        8,
        8,
    )
    .unwrap();
    assert!(index.depth >= 2, "{index:?}");
    let heap = FractalHeapHeader::parse(
        &bytes,
        usize::try_from(info.fractal_heap_address.unwrap().get()).unwrap(),
        8,
        8,
    )
    .unwrap();
    assert!(heap.current_rows_in_root_indirect_block > 0, "{heap:?}");
    dense_links::assert_collision_edit(&path, group_path);
    dir.close().unwrap();
}

#[rstest]
fn editing_a_descendant_relocates_targets_through_dense_ancestors(
    #[values(false, true)] source_only: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ancestors.h5");
    dense_links::write_descendant_edit_fixture(&path);
    let file = group_inspection::open_group_inspection(&path, source_only).unwrap();
    file.group("child").unwrap().create_group("nested").unwrap();
    file.dataset("child/link_0005")
        .unwrap()
        .set_attr("new", crate::AttrValue::I64(99))
        .unwrap();
    file.copy("child/link_0006", "child/copied").unwrap();
    file.commit().unwrap();
    drop(file);
    dense_links::assert_descendant_edit(&path);
    dir.close().unwrap();
}

#[rstest]
fn dense_link_and_attribute_storage_remain_independent(
    #[values(false, true)] source_only: bool,
    #[values(false, true)] edit_attributes: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("attributes.h5");
    dense_links::write_attribute_edit_fixture(&path);
    let file = group_inspection::open_group_inspection(&path, source_only).unwrap();
    let group = file.group("g").unwrap();
    group.create_group("added").unwrap();
    if edit_attributes {
        group.set_attr("attr_0", crate::AttrValue::I64(99)).unwrap();
    }
    file.commit().unwrap();
    drop(group);
    drop(file);
    dense_links::assert_attribute_edit(&path, edit_attributes);
    dir.close().unwrap();
}

#[rstest]
fn unsupported_dense_conversion_rejects_the_batch_before_writing(
    #[values(false, true)] source_only: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unsupported.h5");
    dense_links::write_edit_fixture(&path, "/", 8, None);
    let original = std::fs::read(&path).unwrap();
    let file = group_inspection::open_group_inspection(&path, source_only).unwrap();
    file.root().create_group(&"x".repeat(5000)).unwrap();
    let error = file.commit().unwrap_err();
    assert!(
        matches!(
            error,
            crate::Error::EditUnsupported(
                "a group's resulting links cannot be encoded in dense storage"
            )
        ),
        "{error:?}"
    );
    assert_eq!(file.group_inspection_snapshot().unwrap(), original);
    assert_eq!(file.page_publication_snapshot().unwrap(), original);
    drop(file);
    assert_eq!(std::fs::read(&path).unwrap(), original);
    dir.close().unwrap();
}

#[rstest]
#[case::group(false)]
#[case::dataset(true)]
fn dense_mutation_retains_single_hard_link_relocation_constraints(
    #[case] dataset: bool,
    #[values(false, true)] source_only: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("alias.h5");
    dense_links::write_alias_edit_fixture(&path, dataset);
    let original = std::fs::read(&path).unwrap();
    let file = group_inspection::open_group_inspection(&path, source_only).unwrap();
    if dataset {
        file.dataset("g/link_0000")
            .unwrap()
            .set_attr("new", crate::AttrValue::I64(1))
            .unwrap();
    } else {
        file.group("g").unwrap().create_group("added").unwrap();
    }
    let error = file.commit().unwrap_err();
    let crate::Error::EditUnsupported(reason) = error else {
        panic!("expected EditUnsupported, got {error:?}")
    };
    assert_eq!(
        reason,
        if dataset {
            "editing a dataset attribute relocates its header; only supported when it has a single hard link"
        } else {
            "editing a group relocates its object header; only supported when it has a single hard link"
        }
    );
    assert_eq!(file.group_inspection_snapshot().unwrap(), original);
    drop(file);
    assert_eq!(std::fs::read(&path).unwrap(), original);
    dir.close().unwrap();
}

#[rstest]
fn symbolic_targets_and_name_encoding_survive_both_storage_transitions(
    #[values(false, true)] source_only: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("symbolic.h5");
    dense_links::write_symbolic_edit_fixture(&path);
    let before = group_inspection::inspect_group_links(&path, "/", source_only).unwrap();
    let file = group_inspection::open_group_inspection(&path, source_only).unwrap();
    for index in 0..9 {
        file.root().delete(&format!("link_{index:04}")).unwrap();
    }
    file.commit().unwrap();
    drop(file);
    let compact = group_inspection::inspect_group_links(&path, "/", source_only).unwrap();
    assert!(compact.links.info().fractal_heap_address.is_none());
    assert_symbolic_targets_and_encoding(&compact.links);
    let mut expected = before
        .links
        .links()
        .iter()
        .filter(|link| !link.name().starts_with(b"link_"))
        .cloned()
        .collect::<Vec<_>>();
    expected.sort_by(|a, b| a.name().cmp(b.name()));
    let mut actual = compact.links.links().to_vec();
    actual.sort_by(|a, b| a.name().cmp(b.name()));
    assert_eq!(actual, expected);
    dense_links::assert_compact_symbolic_edit(&path);
    let file = group_inspection::open_group_inspection(&path, source_only).unwrap();
    for index in 0..6 {
        file.root().create_group(&format!("added_{index}")).unwrap();
    }
    file.commit().unwrap();
    drop(file);
    let dense = group_inspection::inspect_group_links(&path, "/", source_only).unwrap();
    assert!(dense.links.info().fractal_heap_address.is_some());
    assert_symbolic_targets_and_encoding(&dense.links);
    let mut actual = dense
        .links
        .links()
        .iter()
        .filter(|link| !link.name().starts_with(b"added_"))
        .cloned()
        .collect::<Vec<_>>();
    actual.sort_by(|a, b| a.name().cmp(b.name()));
    assert_eq!(actual, expected);
    dense_links::assert_dense_symbolic_edit(&path);
    dir.close().unwrap();
}

#[rstest]
fn dense_root_mutation_resolves_stored_addresses_in_a_userblock_file(
    #[values(false, true)] source_only: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("userblock.h5");
    dense_links::write_inspection_fixture(&path, dense_links::GroupOrder::Untracked);
    let before = group_inspection::inspect_group_links(&path, "/", source_only).unwrap();
    assert!(before.links.info().fractal_heap_address.is_some());
    let file = group_inspection::open_group_inspection(&path, source_only).unwrap();
    file.root().create_group("added").unwrap();
    file.commit().unwrap();
    drop(file);
    let after = group_inspection::inspect_group_links(&path, "/", source_only).unwrap();
    assert_eq!(after.links.links().len(), before.links.links().len() + 1);
    let mut actual = after
        .links
        .links()
        .iter()
        .filter(|link| link.name() != b"added")
        .cloned()
        .collect::<Vec<_>>();
    let mut expected = before.links.links().to_vec();
    actual.sort_by(|a, b| a.name().cmp(b.name()));
    expected.sort_by(|a, b| a.name().cmp(b.name()));
    assert_eq!(actual, expected);
    dense_links::assert_userblock_edit(&path);
    dir.close().unwrap();
}

fn assert_symbolic_targets_and_encoding(links: &GroupLinkStorage) {
    let soft = links
        .links()
        .iter()
        .find(|link| link.name() == b"soft")
        .unwrap();
    assert_eq!(
        soft.target(),
        &OwnedLinkTarget::Soft {
            target_path: "/absent_λ".as_bytes().to_vec()
        }
    );
    assert_eq!(soft.charset(), CharacterSet::Ascii);
    let external = links
        .links()
        .iter()
        .find(|link| link.name() == b"external")
        .unwrap();
    assert_eq!(
        external.target(),
        &OwnedLinkTarget::External {
            filename: "absent_λ.h5".as_bytes().to_vec(),
            object_path: "/目标".as_bytes().to_vec(),
        }
    );
    assert_eq!(external.charset(), CharacterSet::Ascii);
    let unicode = links
        .links()
        .iter()
        .find(|link| link.name() == "原始_λ".as_bytes())
        .unwrap();
    assert_eq!(unicode.charset(), CharacterSet::Utf8);
}
