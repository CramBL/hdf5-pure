#![cfg(feature = "__hdf5-1.10")]

use hdf5::File as CFile;
use hdf5::GroupStorageType;
use hdf5::plist::link_create::CharEncoding;
use hdf5_pure::__private;
use hdf5_pure::CharacterSet;
use hdf5_pure_format::__private::BTreeV2Header;
use hdf5_pure_format::__private::FractalHeapHeader;
use hdf5_pure_format::__private::GroupLinkStorage;
use hdf5_pure_format::__private::OwnedLinkTarget;
use rstest::rstest;
use test_util_hdf5::dense_links;

#[rstest]
#[case::compact_boundary(8, 1, 0, None, GroupStorageType::Dense)]
#[case::compact_retained(8, 1, 1, None, GroupStorageType::Compact)]
#[case::dense_boundary(9, 0, 3, None, GroupStorageType::Dense)]
#[case::dense_to_compact(9, 0, 4, None, GroupStorageType::Compact)]
#[case::custom_compact_boundary(12, 1, 0, Some((12, 4)), GroupStorageType::Dense)]
#[case::custom_dense_boundary(13, 0, 9, Some((12, 4)), GroupStorageType::Dense)]
#[case::custom_dense_to_compact(13, 0, 10, Some((12, 4)), GroupStorageType::Compact)]
#[case::empty_compact(9, 0, 9, None, GroupStorageType::Compact)]
#[case::empty_dense(1, 0, 1, Some((0, 0)), GroupStorageType::Dense)]
fn dense_link_edits_select_storage_from_the_final_count(
    #[case] count: usize,
    #[case] added: usize,
    #[case] removed: usize,
    #[case] thresholds: Option<(u32, u32)>,
    #[case] storage: GroupStorageType,
    #[values("/", "/outer/g")] group_path: &str,
    #[values(false, true)] source_only: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("links.h5");
    dense_links::write_edit_fixture(&path, group_path, count, thresholds);
    let file = __private::open_group_inspection(&path, source_only).unwrap();
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
    let reference = CFile::open(&path).unwrap();
    let group = reference.group(group_path).unwrap();
    assert_eq!(group.info().unwrap().storage_type, storage);
    let mut expected = (removed..count)
        .map(|index| format!("link_{index:04}"))
        .collect::<Vec<_>>();
    expected.extend((0..added).map(|index| format!("added_{index}")));
    expected.sort();
    let mut actual = group.member_names().unwrap();
    actual.sort();
    assert_eq!(actual, expected);
    for index in removed..count {
        assert_eq!(
            group
                .dataset(&format!("link_{index:04}"))
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            [i32::try_from(index).unwrap(), 42]
        );
    }
    drop(group);
    reference.close().unwrap();
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
    dense_links::write_edit_fixture(&path, group_path, 2000, None);
    {
        let reference = CFile::open_rw(&path).unwrap();
        let group = reference.group(group_path).unwrap();
        group.link_soft("/absent_λ", "soft").unwrap();
        group
            .link_external("absent_λ.h5", "/目标", "external")
            .unwrap();
        group
            .new_dataset::<i32>()
            .shape(1)
            .create("k69209")
            .unwrap()
            .write(&[7])
            .unwrap();
        group
            .new_dataset::<i32>()
            .shape(1)
            .create("k155448")
            .unwrap()
            .write(&[9])
            .unwrap();
        group
            .create_group_builder()
            .char_encoding(CharEncoding::Utf8)
            .create("原始_λ")
            .unwrap();
        drop(group);
        reference.close().unwrap();
    }
    let before = __private::inspect_group_links(&path, group_path, source_only).unwrap();
    let file = __private::open_group_inspection(&path, source_only).unwrap();
    let group = file.group(group_path).unwrap();
    group.delete("link_0000").unwrap();
    group.create_group("added_λ").unwrap();
    group
        .set_attr("changed", hdf5_pure::AttrValue::I64(17))
        .unwrap();
    file.commit().unwrap();
    drop(group);
    drop(file);
    let after = __private::inspect_group_links(&path, group_path, source_only).unwrap();
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
    let reference = CFile::open(&path).unwrap();
    let group = reference.group(group_path).unwrap();
    assert_eq!(group.info().unwrap().storage_type, GroupStorageType::Dense);
    assert_eq!(group.info().unwrap().nlinks, 2005);
    assert_eq!(
        group.dataset("k69209").unwrap().read_raw::<i32>().unwrap(),
        [7]
    );
    assert_eq!(
        group.dataset("k155448").unwrap().read_raw::<i32>().unwrap(),
        [9]
    );
    assert_eq!(
        group
            .dataset("link_1999")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [1999, 42]
    );
    assert_eq!(
        group.attr("changed").unwrap().read_scalar::<i64>().unwrap(),
        17
    );
    drop(group);
    reference.close().unwrap();
    dir.close().unwrap();
}

#[rstest]
fn editing_a_descendant_relocates_targets_through_dense_ancestors(
    #[values(false, true)] source_only: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ancestors.h5");
    dense_links::write_edit_fixture(&path, "/", 12, None);
    {
        let reference = CFile::open_rw(&path).unwrap();
        let child = reference.create_group("child").unwrap();
        dense_links::populate_edit_group(&child, 12);
        drop(child);
        reference.close().unwrap();
    }
    let file = __private::open_group_inspection(&path, source_only).unwrap();
    file.group("child").unwrap().create_group("nested").unwrap();
    file.dataset("child/link_0005")
        .unwrap()
        .set_attr("new", hdf5_pure::AttrValue::I64(99))
        .unwrap();
    file.copy("child/link_0006", "child/copied").unwrap();
    file.commit().unwrap();
    drop(file);
    let reference = CFile::open(&path).unwrap();
    assert_eq!(
        reference.info().unwrap().storage_type,
        GroupStorageType::Dense
    );
    let child = reference.group("child").unwrap();
    assert_eq!(child.info().unwrap().storage_type, GroupStorageType::Dense);
    assert_eq!(child.info().unwrap().nlinks, 14);
    assert_eq!(
        child
            .dataset("link_0005")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [5, 42]
    );
    assert_eq!(
        child
            .dataset("link_0005")
            .unwrap()
            .attr("new")
            .unwrap()
            .read_scalar::<i64>()
            .unwrap(),
        99
    );
    assert_eq!(
        child.dataset("copied").unwrap().read_raw::<i32>().unwrap(),
        [6, 42]
    );
    assert_eq!(child.group("nested").unwrap().info().unwrap().nlinks, 0);
    assert_eq!(
        reference
            .dataset("link_0011")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [11, 42]
    );
    drop(child);
    reference.close().unwrap();
    dir.close().unwrap();
}

#[rstest]
fn dense_link_and_attribute_storage_remain_independent(
    #[values(false, true)] source_only: bool,
    #[values(false, true)] edit_attributes: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("attributes.h5");
    dense_links::write_edit_fixture(&path, "/g", 12, None);
    {
        let reference = CFile::open_rw(&path).unwrap();
        let group = reference.group("g").unwrap();
        for index in 0..12i64 {
            group
                .new_attr::<i64>()
                .create(format!("attr_{index}").as_str())
                .unwrap()
                .write_scalar(&index)
                .unwrap();
        }
        drop(group);
        reference.close().unwrap();
    }
    let file = __private::open_group_inspection(&path, source_only).unwrap();
    let group = file.group("g").unwrap();
    group.create_group("added").unwrap();
    if edit_attributes {
        group
            .set_attr("attr_0", hdf5_pure::AttrValue::I64(99))
            .unwrap();
    }
    file.commit().unwrap();
    drop(group);
    drop(file);
    let reference = CFile::open(&path).unwrap();
    let group = reference.group("g").unwrap();
    assert_eq!(group.info().unwrap().storage_type, GroupStorageType::Dense);
    assert_eq!(group.info().unwrap().nlinks, 13);
    assert_eq!(group.attr_names().unwrap().len(), 12);
    for index in 0..12i64 {
        assert_eq!(
            group
                .attr(&format!("attr_{index}"))
                .unwrap()
                .read_scalar::<i64>()
                .unwrap(),
            if edit_attributes && index == 0 {
                99
            } else {
                index
            }
        );
    }
    drop(group);
    reference.close().unwrap();
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
    let file = __private::open_group_inspection(&path, source_only).unwrap();
    file.root().create_group(&"x".repeat(5000)).unwrap();
    let error = file.commit().unwrap_err();
    assert!(
        matches!(
            error,
            hdf5_pure::Error::EditUnsupported(
                "a group's resulting links cannot be encoded in dense storage"
            )
        ),
        "{error:?}"
    );
    assert_eq!(
        __private::group_inspection_snapshot(&file).unwrap(),
        original
    );
    assert_eq!(
        __private::page_publication_snapshot(&file).unwrap(),
        original
    );
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
    dense_links::write_edit_fixture(&path, "/g", 12, None);
    {
        let reference = CFile::open_rw(&path).unwrap();
        reference
            .link_hard(if dataset { "/g/link_0000" } else { "/g" }, "alias")
            .unwrap();
        reference.close().unwrap();
    }
    let original = std::fs::read(&path).unwrap();
    let file = __private::open_group_inspection(&path, source_only).unwrap();
    if dataset {
        file.dataset("g/link_0000")
            .unwrap()
            .set_attr("new", hdf5_pure::AttrValue::I64(1))
            .unwrap();
    } else {
        file.group("g").unwrap().create_group("added").unwrap();
    }
    let error = file.commit().unwrap_err();
    let hdf5_pure::Error::EditUnsupported(reason) = error else {
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
    assert_eq!(
        __private::group_inspection_snapshot(&file).unwrap(),
        original
    );
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
    dense_links::write_edit_fixture(&path, "/", 9, None);
    {
        let reference = CFile::open_rw(&path).unwrap();
        reference.link_soft("/absent_λ", "soft").unwrap();
        reference
            .link_external("absent_λ.h5", "/目标", "external")
            .unwrap();
        reference
            .create_group_builder()
            .char_encoding(CharEncoding::Utf8)
            .create("原始_λ")
            .unwrap();
        reference.close().unwrap();
    }
    let before = __private::inspect_group_links(&path, "/", source_only).unwrap();
    let file = __private::open_group_inspection(&path, source_only).unwrap();
    for index in 0..9 {
        file.root().delete(&format!("link_{index:04}")).unwrap();
    }
    file.commit().unwrap();
    drop(file);
    let compact = __private::inspect_group_links(&path, "/", source_only).unwrap();
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
    let reference = CFile::open(&path).unwrap();
    assert_eq!(
        reference.info().unwrap().storage_type,
        GroupStorageType::Compact
    );
    let mut names = reference.member_names().unwrap();
    names.sort();
    assert_eq!(names, ["external", "soft", "原始_λ"]);
    reference.close().unwrap();
    let file = __private::open_group_inspection(&path, source_only).unwrap();
    for index in 0..6 {
        file.root().create_group(&format!("added_{index}")).unwrap();
    }
    file.commit().unwrap();
    drop(file);
    let dense = __private::inspect_group_links(&path, "/", source_only).unwrap();
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
    let reference = CFile::open(&path).unwrap();
    assert_eq!(
        reference.info().unwrap().storage_type,
        GroupStorageType::Dense
    );
    assert_eq!(reference.info().unwrap().nlinks, 9);
    reference.close().unwrap();
    dir.close().unwrap();
}

#[rstest]
fn dense_root_mutation_resolves_stored_addresses_in_a_userblock_file(
    #[values(false, true)] source_only: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("userblock.h5");
    test_util_hdf5::dense_links::write_inspection_fixture(
        &path,
        "untracked",
        hdf5::plist::group_create::LinkCreationOrder::Untracked,
    );
    let before = __private::inspect_group_links(&path, "/", source_only).unwrap();
    assert!(before.links.info().fractal_heap_address.is_some());
    let file = __private::open_group_inspection(&path, source_only).unwrap();
    file.root().create_group("added").unwrap();
    file.commit().unwrap();
    drop(file);
    let after = __private::inspect_group_links(&path, "/", source_only).unwrap();
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
    let reference = CFile::open(&path).unwrap();
    assert_eq!(
        reference.info().unwrap().storage_type,
        GroupStorageType::Dense
    );
    assert_eq!(reference.group("added").unwrap().info().unwrap().nlinks, 0);
    assert_eq!(
        reference
            .group("dense_untracked")
            .unwrap()
            .info()
            .unwrap()
            .nlinks,
        2000
    );
    reference.close().unwrap();
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
