#![cfg(feature = "__hdf5-1.10")]

use std::collections::BTreeMap;

use hdf5::File as CFile;
use hdf5::GroupStorageType;
use hdf5::plist::group_create::LinkCreationOrder;
use hdf5_pure::__private;
use hdf5_pure::CharacterSet;
use hdf5_pure::Error;
use hdf5_pure_format::__private::BTreeV2Header;
use hdf5_pure_format::__private::FractalHeapHeader;
use hdf5_pure_format::__private::GroupPhaseChange;
use hdf5_pure_format::__private::OwnedLinkTarget;
use rstest::rstest;

#[rstest]
#[case::untracked("untracked", LinkCreationOrder::Untracked, false, false)]
#[case::tracked("tracked", LinkCreationOrder::Tracked, true, false)]
#[case::indexed("indexed", LinkCreationOrder::Indexed, true, true)]
fn libhdf5_groups_retain_links_policy_and_thresholds(
    #[case] policy: &str,
    #[case] order: LinkCreationOrder,
    #[case] tracked: bool,
    #[case] indexed: bool,
    #[values(("compact", 6), ("dense", 2000))] form: (&str, usize),
    #[values(false, true)] source_only: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("group.h5");
    test_util_hdf5::dense_links::write_inspection_fixture(&path, policy, order);
    let group_path = format!("{}_{policy}", form.0);
    let info = __private::inspect_group_links(&path, &group_path, source_only).unwrap();
    assert_eq!(info.links.links().len(), form.1);
    let metadata = info.links.info();
    assert_eq!(
        metadata.max_creation_order,
        tracked.then_some(form.1 as u64 + 1)
    );
    assert_eq!(metadata.btree_creation_order_address.is_some(), indexed);
    assert_eq!(metadata.fractal_heap_address.is_some(), form.0 == "dense");
    let links = info
        .links
        .links()
        .iter()
        .map(|link| (link.name(), link))
        .collect::<BTreeMap<_, _>>();
    let expected_index = |index| tracked.then_some(index);
    let hard = links[b"hard".as_slice()];
    let target = __private::inspect_group_links(&path, "target", source_only).unwrap();
    assert_eq!(
        hard.target(),
        &OwnedLinkTarget::Hard {
            object_header_address: target.object_header_address
        }
    );
    assert_eq!(hard.creation_order(), expected_index(1));
    assert_eq!(
        links[b"soft".as_slice()].target(),
        &OwnedLinkTarget::Soft {
            target_path: b"/target".to_vec()
        }
    );
    assert_eq!(
        links[b"soft".as_slice()].creation_order(),
        expected_index(2)
    );
    assert_eq!(
        links[b"external".as_slice()].target(),
        &OwnedLinkTarget::External {
            filename: "外部.h5".as_bytes().to_vec(),
            object_path: "/目标".as_bytes().to_vec()
        }
    );
    assert_eq!(
        links[b"external".as_slice()].creation_order(),
        expected_index(3)
    );
    assert_eq!(links["utf8_λ".as_bytes()].charset(), CharacterSet::Utf8);
    assert_eq!(
        links["utf8_λ".as_bytes()].creation_order(),
        expected_index(4)
    );
    assert_eq!(links[b"k69209".as_slice()].target(), hard.target());
    assert_eq!(links[b"k155448".as_slice()].target(), hard.target());
    assert_eq!(info.group_info, [0, 1, 12, 0, 4, 0]);
    let thresholds = GroupPhaseChange::parse(&info.group_info).unwrap();
    assert_eq!(thresholds.max_compact(), 12);
    assert_eq!(thresholds.min_dense(), 4);
    if form.0 == "dense" {
        let bytes = std::fs::read(&path).unwrap();
        let header = BTreeV2Header::parse(
            &bytes,
            512 + usize::try_from(metadata.btree_name_index_address.unwrap().get()).unwrap(),
            8,
            8,
        )
        .unwrap();
        assert!(header.depth >= 2, "{header:?}");
        let heap = FractalHeapHeader::parse(
            &bytes,
            512 + usize::try_from(metadata.fractal_heap_address.unwrap().get()).unwrap(),
            8,
            8,
        )
        .unwrap();
        assert!(heap.current_rows_in_root_indirect_block > 0, "{heap:?}");
    }
    dir.close().unwrap();
}

#[rstest]
#[case::root("")]
#[case::dense_child("dense_indexed")]
#[case::compact_child("compact_indexed")]
fn dense_root_edits_preserve_untracked_and_tracked_group_policies(
    #[case] parent: &str,
    #[values(false, true)] source_only: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("group.h5");
    test_util_hdf5::dense_links::write_inspection_fixture(
        &path,
        "indexed",
        LinkCreationOrder::Indexed,
    );
    let before = __private::inspect_group_links(&path, parent, source_only).unwrap();
    let file = __private::open_group_inspection(&path, source_only).unwrap();
    file.group(parent).unwrap().create_group("added").unwrap();
    file.commit().unwrap();
    drop(file);
    let after = __private::inspect_group_links(&path, parent, source_only).unwrap();
    assert_eq!(after.links.links().len(), before.links.links().len() + 1);
    assert_eq!(
        after.links.info().max_creation_order,
        before
            .links
            .info()
            .max_creation_order
            .map(|counter| counter + 1)
    );
    let reference = CFile::open(&path).unwrap();
    let group = reference.group(&format!("/{parent}/added")).unwrap();
    assert_eq!(group.info().unwrap().nlinks, 0);
    drop(group);
    reference.close().unwrap();
    dir.close().unwrap();
}

#[rstest]
#[case::filtered(
    false,
    "filtered",
    "a group's fractal heap uses filters, huge objects, or tiny objects"
)]
#[case::huge(
    true,
    "g",
    "a group's fractal heap uses filters, huge objects, or tiny objects"
)]
fn unsupported_heap_preflight_rejects_mutation_before_writing(
    #[case] huge_objects: bool,
    #[case] group_path: &str,
    #[case] reason: &str,
    #[values(false, true)] source_only: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("group.h5");
    if huge_objects {
        test_util_hdf5::dense_links::write_huge_inspection_fixture(&path);
    } else {
        test_util_hdf5::dense_links::write_inspection_fixture(
            &path,
            "indexed",
            LinkCreationOrder::Indexed,
        );
    }
    let original = std::fs::read(&path).unwrap();
    let error = __private::inspect_group_links(&path, group_path, source_only)
        .err()
        .unwrap();
    assert!(
        matches!(
            error,
            Error::EditUnsupported(
                "a group's fractal heap uses filters, huge objects, or tiny objects"
            )
        ),
        "{error:?}"
    );
    let file = __private::open_group_inspection(&path, source_only).unwrap();
    file.group(group_path)
        .unwrap()
        .create_group("added")
        .unwrap();
    let error = file.commit().unwrap_err();
    assert!(
        matches!(
            error,
            Error::EditUnsupported(message) if message == reason
        ),
        "{error:?}"
    );
    assert_eq!(
        __private::group_inspection_snapshot(&file).unwrap(),
        original
    );
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        u64::try_from(original.len()).unwrap()
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
#[case::untracked("untracked", LinkCreationOrder::Untracked)]
#[case::tracked("tracked", LinkCreationOrder::Tracked)]
#[case::indexed("indexed", LinkCreationOrder::Indexed)]
fn generated_groups_have_expected_storage_policy_and_links(
    #[case] policy: &str,
    #[case] order: LinkCreationOrder,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("group.h5");
    test_util_hdf5::dense_links::write_inspection_fixture(&path, policy, order);
    let file = CFile::open(&path).unwrap();
    assert_eq!(file.info().unwrap().storage_type, GroupStorageType::Dense);
    test_util_hdf5::dense_links::assert_inspection_group(
        &file,
        &format!("compact_{policy}"),
        6,
        GroupStorageType::Compact,
        order,
    );
    test_util_hdf5::dense_links::assert_inspection_group(
        &file,
        &format!("dense_{policy}"),
        2000,
        GroupStorageType::Dense,
        order,
    );
    assert_eq!(
        file.group("filtered").unwrap().info().unwrap().storage_type,
        GroupStorageType::Dense
    );
    file.close().unwrap();
    dir.close().unwrap();
}
