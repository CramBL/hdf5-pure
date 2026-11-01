//! Native observations of link and attribute metadata before and after repacking.

use std::path::Path;

use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::GroupLinkStorage;
use hdf5_pure_format::__private::OwnedLink;
use hdf5_pure_format::__private::OwnedLinkTarget;
use rstest::rstest;
use test_util_hdf5::creation_order;
use test_util_hdf5::creation_order::Indexing;
use test_util_hdf5::dense_links;
use test_util_hdf5::dense_links::GroupOrder;
use test_util_hdf5::exotic_attributes;
use test_util_hdf5::file;

use crate::Error;
use crate::File;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::object_header::ObjectHeader;
use crate::width::OffsetWidth;

use super::test_util::ExoticAttributes;
use crate::RepackOptions;

#[rstest]
#[case::compact_empty(0, false, "a compact group's link filters cannot be reproduced")]
#[case::compact(6, false, "a compact group's link filters cannot be reproduced")]
#[case::dense(
    16,
    false,
    "a group's fractal heap uses filters, huge objects, or tiny objects"
)]
#[case::huge(
    12,
    true,
    "a group's fractal heap uses filters, huge objects, or tiny objects"
)]
fn repack_rejects_unsupported_link_storage_before_publication_and_can_drop_it(
    #[case] count: usize,
    #[case] huge: bool,
    #[case] reason: &str,
    #[values(false, true)] streaming: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("source.h5");
    let destination = directory.path().join("repacked.h5");
    let group_path = if huge {
        dense_links::write_huge_inspection_fixture(&path);
        "g"
    } else {
        dense_links::write_filtered_link_fixture(&path, count);
        "filtered"
    };
    let source = if streaming {
        File::open_streaming(&path)
    } else {
        File::open(&path)
    }
    .unwrap();
    let header = source.group(group_path).unwrap().header_address().unwrap();
    let header = ObjectHeader::parse(
        &std::fs::read(&path).unwrap(),
        source.access_mode(),
        usize::try_from(header).unwrap(),
        8,
        8,
    )
    .unwrap();
    assert_eq!(
        header
            .messages
            .iter()
            .any(|message| message.msg_type == crate::MessageType::LINK),
        !huge && count != 0 && count <= 8
    );
    let mut sink = Vec::new();
    let error = super::repack_into(&source, &mut sink, &RepackOptions::new()).unwrap_err();
    assert!(
        matches!(error, Error::RepackUnsupported(ref message) if message == reason),
        "{error:?}"
    );
    assert!(sink.is_empty());
    let error = super::repack_to_path(&source, &destination, &RepackOptions::new()).unwrap_err();
    assert!(
        matches!(error, Error::RepackUnsupported(ref message) if message == reason),
        "{error:?}"
    );
    assert!(!destination.exists());
    super::repack_to_path(
        &source,
        &destination,
        &RepackOptions::new().drop_path(group_path),
    )
    .unwrap();
    source.close().unwrap();
    let native = file::libhdf5_open(&destination);
    assert_eq!(native.member_names().unwrap(), Vec::<String>::new());
    assert_eq!(native.info().unwrap().nlinks, 0);
    native.close().unwrap();
    directory.close().unwrap();
}

#[rstest]
#[case::ordinary((12, 8))]
#[case::one_way((12, 0))]
#[case::always_dense((0, 0))]
fn reconstructed_link_policies_keep_counters_across_edits_and_empty_reconstruction(
    #[case] thresholds: (u32, u32),
    #[values(Indexing::No, Indexing::Yes)] indexing: Indexing,
    #[values("/", "/group")] group_path: &str,
    #[values(FileSpaceStrategy::FsmAggr, FileSpaceStrategy::Page)] strategy: FileSpaceStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("source.h5");
    let destination = directory.path().join("repacked.h5");
    let empty = directory.path().join("empty.h5");
    creation_order::write_link_fixture(&path, group_path, 13, indexing, thresholds);
    let source = File::open_rw(&path).unwrap();
    source
        .group(group_path)
        .unwrap()
        .delete("link_0000")
        .unwrap();
    source.commit().unwrap();
    let expected_policy = source
        .group(group_path)
        .unwrap()
        .link_phase_change()
        .unwrap();
    let options = RepackOptions::new()
        .with_file_space_settings(FileSpaceSettings::new(strategy).with_persistence(true));
    super::repack_to_path(&source, &destination, &options).unwrap();
    source.close().unwrap();
    let file = File::open(&destination).unwrap();
    let group = file.group(group_path).unwrap();
    assert_eq!(group.link_phase_change().unwrap(), expected_policy);
    let mut expected = (1..13)
        .map(|index| (format!("link_{index:04}"), index))
        .collect::<Vec<_>>();
    drop(group);
    file.close().unwrap();
    assert_tracked_links(
        &destination,
        group_path,
        &expected,
        13,
        thresholds.0 == 0,
        indexing,
    );
    let file = File::open_rw(&destination).unwrap();
    let group = file.group(group_path).unwrap();
    group.create_group("added").unwrap();
    file.commit().unwrap();
    expected.push(("added".into(), 13));
    drop(group);
    file.close().unwrap();
    assert_tracked_links(&destination, group_path, &expected, 14, true, indexing);
    let file = File::open_rw(&destination).unwrap();
    let group = file.group(group_path).unwrap();
    for index in 1..6 {
        group.delete(&format!("link_{index:04}")).unwrap();
    }
    file.commit().unwrap();
    expected.retain(|(_, index)| *index >= 6);
    drop(group);
    file.close().unwrap();
    assert_tracked_links(&destination, group_path, &expected, 14, true, indexing);
    let file = File::open_rw(&destination).unwrap();
    let group = file.group(group_path).unwrap();
    group.delete("link_0006").unwrap();
    file.commit().unwrap();
    expected.retain(|(_, index)| *index >= 7);
    drop(group);
    file.close().unwrap();
    assert_tracked_links(
        &destination,
        group_path,
        &expected,
        14,
        thresholds.1 == 0,
        indexing,
    );
    let file = File::open_rw(&destination).unwrap();
    let group = file.group(group_path).unwrap();
    for (name, _) in &expected {
        group.delete(name).unwrap();
    }
    file.commit().unwrap();
    drop(group);
    file.close().unwrap();
    assert_tracked_links(
        &destination,
        group_path,
        &[],
        14,
        thresholds.1 == 0,
        indexing,
    );
    let file = File::open(&destination).unwrap();
    super::repack_to_path(&file, &empty, &options).unwrap();
    file.close().unwrap();
    let file = File::open(&empty).unwrap();
    assert_eq!(
        file.group(group_path).unwrap().link_phase_change().unwrap(),
        expected_policy
    );
    file.close().unwrap();
    assert_tracked_links(&empty, group_path, &[], 14, false, indexing);
    let native = file::libhdf5_open(&empty);
    let group = native.group(group_path).unwrap();
    let policy = group.gcpl().unwrap().get_link_phase_change().unwrap();
    assert_eq!((policy.max_compact, policy.min_dense), thresholds);
    assert_eq!(group.info().unwrap().max_corder, 14);
    assert_eq!(group.info().unwrap().nlinks, 0);
    drop(group);
    native.close().unwrap();
    creation_order::add_reference_group(&empty, &format!("{group_path}/native"));
    assert_tracked_links(
        &empty,
        group_path,
        &[("native".into(), 14)],
        15,
        thresholds.0 == 0,
        indexing,
    );
    let native = file::libhdf5_open(&empty);
    let group = native.group(group_path).unwrap();
    assert_eq!(group.info().unwrap().max_corder, 15);
    assert_eq!(dense_links::creation_order_names(&group), ["native"]);
    drop(group);
    native.close().unwrap();
    directory.close().unwrap();
}

fn assert_tracked_links(
    file_path: &Path,
    path: &str,
    expected: &[(String, u64)],
    next: u64,
    dense: bool,
    indexing: Indexing,
) {
    let file = File::open(file_path).unwrap();
    let links = read_links(&file, path);
    assert_eq!(links.creation_order().next_index(), Some(next));
    assert_eq!(links.info().fractal_heap_address.is_some(), dense);
    assert_eq!(links.info().btree_name_index_address.is_some(), dense);
    assert_eq!(
        links.info().btree_creation_order_address.is_some(),
        matches!(indexing, Indexing::Yes)
    );
    assert_eq!(
        links
            .info()
            .btree_creation_order_address
            .flatten()
            .is_some(),
        dense && matches!(indexing, Indexing::Yes)
    );
    let mut observed = links
        .links()
        .iter()
        .map(|link| {
            (
                String::from_utf8(link.name().to_vec()).unwrap(),
                link.creation_order().unwrap(),
            )
        })
        .collect::<Vec<_>>();
    observed.sort_by_key(|(_, index)| *index);
    assert_eq!(observed, expected);
    file.close().unwrap();
    creation_order::assert_link_group(file_path, path, expected, next, indexing, dense);
}

#[rstest]
#[case::compact(6)]
#[case::dense(13)]
fn repack_preserves_complete_link_metadata(
    #[case] count: usize,
    #[values(GroupOrder::Untracked, GroupOrder::Tracked, GroupOrder::Indexed)] order: GroupOrder,
    #[values("/", "group")] group_path: &str,
    #[values(false, true)] drop_child: bool,
    #[values(false, true)] streaming: bool,
    #[values(FileSpaceStrategy::FsmAggr, FileSpaceStrategy::Page)] strategy: FileSpaceStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let source_path = directory.path().join("source.h5");
    let destination = directory.path().join("repacked.h5");
    dense_links::write_repack_link_fixture(&source_path, order, group_path, count);
    let source = if streaming {
        File::open_streaming(&source_path)
    } else {
        File::open(&source_path)
    }
    .unwrap();
    let before = read_links(&source, group_path);
    let expected_policy = source
        .group(group_path)
        .unwrap()
        .link_phase_change()
        .unwrap();
    let expected_creation = before.creation_order();
    assert_eq!(
        expected_creation.next_index(),
        if matches!(order, GroupOrder::Untracked) {
            None
        } else {
            Some(u64::try_from(count + 1).unwrap())
        }
    );
    assert_eq!(before.info().fractal_heap_address.is_some(), count > 12);
    let mut expected = normalized_links(before);
    if drop_child {
        expected.retain(|link| link.name() != b"child");
    }
    let options = RepackOptions::new()
        .with_file_space_settings(FileSpaceSettings::new(strategy).with_persistence(true));
    let options = if drop_child {
        options.drop_path(&format!("{group_path}/child"))
    } else {
        options
    };
    super::repack_to_path(&source, &destination, &options).unwrap();
    source.close().unwrap();
    let file = File::open(&destination).unwrap();
    let after = read_links(&file, group_path);
    assert_eq!(after.creation_order(), expected_creation);
    let dense = count - usize::from(drop_child) > 12;
    assert_eq!(after.info().fractal_heap_address.is_some(), dense);
    assert_eq!(after.info().btree_name_index_address.is_some(), dense);
    assert_eq!(
        after.info().btree_creation_order_address.is_some(),
        matches!(order, GroupOrder::Indexed)
    );
    assert_eq!(
        after
            .info()
            .btree_creation_order_address
            .flatten()
            .is_some(),
        dense && matches!(order, GroupOrder::Indexed)
    );
    assert_eq!(normalized_links(after), expected);
    assert_eq!(
        file.group(group_path).unwrap().link_phase_change().unwrap(),
        expected_policy
    );
    assert_eq!(
        file.group(group_path)
            .unwrap()
            .dataset("soft")
            .unwrap()
            .read_i32()
            .unwrap(),
        [7, 17]
    );
    file.close().unwrap();
    let native = file::libhdf5_open(&destination);
    let group = native.group(group_path).unwrap();
    assert_eq!(
        group.gcpl().unwrap().get_link_creation_order().unwrap(),
        order.creation_order()
    );
    assert_eq!(
        group
            .gcpl()
            .unwrap()
            .get_link_phase_change()
            .unwrap()
            .max_compact,
        12
    );
    assert_eq!(
        group.info().unwrap().nlinks,
        u64::try_from(count - usize::from(drop_child)).unwrap()
    );
    assert_eq!(
        group.dataset("value").unwrap().read_raw::<i32>().unwrap(),
        [7, 17]
    );
    assert_eq!(
        group.dataset("soft").unwrap().read_raw::<i32>().unwrap(),
        [7, 17]
    );
    let mut names = group.member_names().unwrap();
    names.sort();
    assert_eq!(
        names,
        expected
            .iter()
            .map(|link| String::from_utf8(link.name().to_vec()).unwrap())
            .collect::<Vec<_>>()
    );
    let mut observed = Vec::new();
    group
        .iter_visit_default(|name, info| {
            observed.push((name.to_owned(), info.creation_order));
            Ok(())
        })
        .unwrap();
    observed.sort();
    assert_eq!(
        observed,
        expected
            .iter()
            .map(|link| (
                String::from_utf8(link.name().to_vec()).unwrap(),
                link.creation_order()
                    .map(|index| i64::try_from(index).unwrap())
            ))
            .collect::<Vec<_>>()
    );
    if let Some(next) = expected_creation.next_index() {
        assert_eq!(
            group.info().unwrap().max_corder,
            i64::try_from(next).unwrap()
        );
        let mut creation = expected.iter().collect::<Vec<_>>();
        creation.sort_by_key(|link| link.creation_order());
        assert_eq!(
            dense_links::creation_order_names(&group),
            creation
                .iter()
                .map(|link| String::from_utf8(link.name().to_vec()).unwrap())
                .collect::<Vec<_>>()
        );
    }
    let mut identities = Vec::new();
    for link in &expected {
        if matches!(link.target(), OwnedLinkTarget::Hard { .. }) {
            let name = std::str::from_utf8(link.name()).unwrap();
            let identity = group.loc_info_by_name(name).unwrap().token;
            assert!(
                !identities.contains(&identity),
                "duplicate object identity for {name:?}"
            );
            identities.push(identity);
        }
    }
    drop(group);
    native.close().unwrap();
    directory.close().unwrap();
}

#[test]
fn the_c_library_reads_fresh_exotic_attributes_before_and_after_repack() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("source.h5");
    let dst = dir.path().join("repacked.h5");
    let exotic = ExoticAttributes::new();
    exotic.write(&src);
    exotic_attributes::assert_libhdf5_contents(&src);
    super::repack(&src, &dst, &RepackOptions::new()).unwrap();
    exotic.assert_repacked(&src, &dst);
    exotic_attributes::assert_libhdf5_contents(&dst);
}

fn read_links(file: &File, path: &str) -> GroupLinkStorage {
    let address = PhysicalAddress::new(file.group(path).unwrap().header_address().unwrap());
    file.repack_links_at(address).unwrap().1.unwrap()
}

fn normalized_links(storage: GroupLinkStorage) -> Vec<OwnedLink> {
    let mut links = storage.into_links();
    for link in &mut links {
        if matches!(link.target(), OwnedLinkTarget::Hard { .. }) {
            link.set_hard_target(StoredAddress::ZERO, OffsetWidth::Eight)
                .unwrap();
        }
    }
    links.sort_by(|a, b| a.name().cmp(b.name()));
    links
}
