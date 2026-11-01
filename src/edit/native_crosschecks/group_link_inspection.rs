use std::collections::BTreeMap;
use std::path::Path;

use hdf5_pure_format::__private::BTreeV2Header;
use hdf5_pure_format::__private::FractalHeapHeader;
use hdf5_pure_format::__private::GroupPhaseChange;
use hdf5_pure_format::__private::LinkInfoMessage;
use hdf5_pure_format::__private::ObjectHeader;
use hdf5_pure_format::__private::OwnedLinkTarget;
use rstest::rstest;
use test_util::filter_pipeline;
use test_util::group_info;
use test_util::group_info::HeaderVersion;
use test_util::group_info::Userblock;
use test_util::object_header::Message;
use test_util::object_header::MessageType as FixtureMessageType;
use test_util_hdf5::creation_order;
use test_util_hdf5::dense_links;
use test_util_hdf5::dense_links::GroupOrder;
use test_util_hdf5::file;

use crate::CharacterSet;
use crate::Error;
use crate::File;
use crate::FileAccessProperties;
use crate::MemoryStrategy;
use crate::MessageType;
use crate::access_mode::AccessMode;
use crate::edit::group_inspection;
use crate::width::OffsetWidth;

#[rstest]
#[case::default((8, 6), 1, false)]
#[case::custom((12, 8), 1, false)]
#[case::staged_custom((12, 8), 2, true)]
#[case::one_way((12, 0), 1, false)]
#[case::always_dense((0, 0), 1, false)]
fn filtered_compact_transitions_reject_mutation_before_writing(
    #[case] thresholds: (u32, u32),
    #[case] added: usize,
    #[case] replace: bool,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] memory: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("edited.h5");
    let reference = directory.path().join("reference.h5");
    let count = usize::try_from(thresholds.0).unwrap();
    dense_links::write_filtered_link_fixture(&path, count - usize::from(replace), thresholds);
    if replace {
        creation_order::add_reference_group(&path, "filtered/replace");
    }
    dense_links::write_filtered_link_fixture(&reference, count, thresholds);
    let before = filtered_group_metadata(&path);
    assert_eq!(before.1, None);
    creation_order::add_reference_group(&reference, "filtered/added");
    let native = filtered_group_metadata(&reference);
    assert_eq!(native.0, before.0);
    assert_eq!(native.1, Some(true));
    let original = std::fs::read(&path).unwrap();
    let session = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(memory),
    )
    .unwrap();
    let group = session.group("filtered").unwrap();
    for index in 0..added {
        group.create_group(&format!("added_{index}")).unwrap();
    }
    if replace {
        group.delete("replace").unwrap();
    }
    let error = session.commit().unwrap_err();
    assert!(
        matches!(
            error,
            Error::EditUnsupported("a group's link filters cannot be reproduced in dense storage")
        ),
        "{error:?}"
    );
    assert!(session.has_staged_edits());
    assert_eq!(session.group_inspection_snapshot().unwrap(), original);
    assert_eq!(session.page_publication_snapshot().unwrap(), original);
    assert_eq!(session.disk_bytes().unwrap(), original);
    drop(group);
    drop(session);
    assert_eq!(std::fs::read(&path).unwrap(), original);
    assert_eq!(filtered_group_metadata(&path), before);
    directory.close().unwrap();
}

#[rstest]
#[case::default((8, 6))]
#[case::custom((12, 8))]
#[case::one_way((12, 0))]
fn filtered_compact_edits_preserve_the_pipeline_and_policy(
    #[case] thresholds: (u32, u32),
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] memory: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("edited.h5");
    let count = usize::try_from(thresholds.0).unwrap();
    dense_links::write_filtered_link_fixture(&path, count - 1, thresholds);
    creation_order::add_reference_group(&path, "filtered/replace");
    let before = filtered_group_metadata(&path);
    assert_eq!(before.1, None);
    let native = file::libhdf5_open(&path);
    let filters = native
        .group("filtered")
        .unwrap()
        .gcpl()
        .unwrap()
        .get_filters()
        .unwrap();
    assert_eq!(filters.len(), 1);
    native.close().unwrap();
    let session = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(memory),
    )
    .unwrap();
    let group = session.group("filtered").unwrap();
    let policy = group.link_phase_change().unwrap();
    group.create_group("added").unwrap();
    group.delete("replace").unwrap();
    session.commit().unwrap();
    assert_eq!(group.link_phase_change().unwrap(), policy);
    drop(group);
    session.close().unwrap();
    assert_eq!(filtered_group_metadata(&path), before);
    let native = file::libhdf5_open(&path);
    let group = native.group("filtered").unwrap();
    assert_eq!(group.gcpl().unwrap().get_filters().unwrap(), filters);
    let policy = group.gcpl().unwrap().get_link_phase_change().unwrap();
    assert_eq!((policy.max_compact, policy.min_dense), thresholds);
    let mut names = group.member_names().unwrap();
    names.sort();
    let expected = std::iter::once("added".to_owned())
        .chain((0..count - 1).map(|index| format!("link_{index:04}")))
        .collect::<Vec<_>>();
    assert_eq!(names, expected);
    assert_eq!(group.info().unwrap().nlinks, u64::try_from(count).unwrap());
    drop(group);
    native.close().unwrap();
    directory.close().unwrap();
}

fn filtered_group_metadata(path: &Path) -> (Vec<u8>, Option<bool>) {
    let file = File::open(path).unwrap();
    let address =
        usize::try_from(file.group("filtered").unwrap().header_address().unwrap()).unwrap();
    file.close().unwrap();
    let bytes = std::fs::read(path).unwrap();
    let header = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, address, 8, 8).unwrap();
    let pipeline = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::FILTER_PIPELINE)
        .unwrap()
        .data
        .clone();
    let info = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::LINK_INFO)
        .unwrap();
    let info = LinkInfoMessage::parse_with_offset_width(&info.data, OffsetWidth::Eight).unwrap();
    let filtered = info.fractal_heap_address.map(|address| {
        FractalHeapHeader::parse(&bytes, usize::try_from(address.get()).unwrap(), 8, 8)
            .unwrap()
            .filtering
            .is_filtered()
    });
    (pipeline, filtered)
}

#[rstest]
#[case::compact(false)]
#[case::dense(true)]
fn version_one_group_conversion_preserves_filters_or_rejects_dense_storage(#[case] dense: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("version-one.h5");
    let pipeline = filter_pipeline::user_filter(true);
    std::fs::write(
        &path,
        group_info::file(
            &[Message::new(FixtureMessageType::FILTER_PIPELINE, &pipeline)],
            HeaderVersion::V1,
            Userblock::Absent,
        ),
    )
    .unwrap();
    let native = file::libhdf5_open(&path);
    let expected = native
        .group("child")
        .unwrap()
        .gcpl()
        .unwrap()
        .get_filters()
        .unwrap();
    assert_eq!(expected.len(), 1);
    native.close().unwrap();
    let original = std::fs::read(&path).unwrap();
    let session = File::open_rw(&path).unwrap();
    let group = session.group("child").unwrap();
    for index in 0..if dense { 9 } else { 1 } {
        group.create_group(&format!("added_{index}")).unwrap();
    }
    if dense {
        let error = session.commit().unwrap_err();
        assert!(
            matches!(
                error,
                Error::EditUnsupported(
                    "a group's link filters cannot be reproduced in dense storage"
                )
            ),
            "{error:?}"
        );
        assert_eq!(session.group_inspection_snapshot().unwrap(), original);
        assert_eq!(session.page_publication_snapshot().unwrap(), original);
    } else {
        session.commit().unwrap();
    }
    drop(group);
    drop(session);
    if dense {
        assert_eq!(std::fs::read(&path).unwrap(), original);
    }
    let native = file::libhdf5_open(&path);
    let group = native.group("child").unwrap();
    assert_eq!(group.gcpl().unwrap().get_filters().unwrap(), expected);
    let expected_names = if dense {
        Vec::new()
    } else {
        vec!["added_0".to_owned()]
    };
    assert_eq!(group.member_names().unwrap(), expected_names);
    drop(group);
    native.close().unwrap();
    directory.close().unwrap();
}

#[rstest]
#[case::untracked(GroupOrder::Untracked, false, false)]
#[case::tracked(GroupOrder::Tracked, true, false)]
#[case::indexed(GroupOrder::Indexed, true, true)]
fn libhdf5_groups_retain_links_policy_and_thresholds(
    #[case] order: GroupOrder,
    #[case] tracked: bool,
    #[case] indexed: bool,
    #[values(("compact", 6), ("dense", 2000))] form: (&str, usize),
    #[values(false, true)] source_only: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("group.h5");
    test_util_hdf5::dense_links::write_inspection_fixture(&path, order);
    let group_path = format!("{}_{}", form.0, order.label());
    let info = group_inspection::inspect_group_links(&path, &group_path, source_only).unwrap();
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
    let target = group_inspection::inspect_group_links(&path, "target", source_only).unwrap();
    assert_eq!(
        hard.target(),
        &OwnedLinkTarget::Hard {
            object_header_address: target.object_header_address
        }
    );
    assert_eq!(hard.creation_order(), expected_index(1));
    assert_eq!(
        links[b"soft".as_slice()].target(),
        &OwnedLinkTarget::soft(b"/target".to_vec())
    );
    assert_eq!(
        links[b"soft".as_slice()].creation_order(),
        expected_index(2)
    );
    assert_eq!(
        links[b"external".as_slice()].target(),
        &OwnedLinkTarget::external("外部.h5".as_bytes().to_vec(), "/目标".as_bytes().to_vec())
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
    test_util_hdf5::dense_links::write_inspection_fixture(&path, GroupOrder::Indexed);
    let before = group_inspection::inspect_group_links(&path, parent, source_only).unwrap();
    let file = group_inspection::open_group_inspection(&path, source_only).unwrap();
    file.group(parent).unwrap().create_group("added").unwrap();
    file.commit().unwrap();
    drop(file);
    let after = group_inspection::inspect_group_links(&path, parent, source_only).unwrap();
    assert_eq!(after.links.links().len(), before.links.links().len() + 1);
    let mut actual = after
        .links
        .links()
        .iter()
        .filter(|link| link.name() != b"added")
        .cloned()
        .collect::<Vec<_>>();
    let mut expected = before.links.links().to_vec();
    actual.sort_by(|left, right| left.name().cmp(right.name()));
    expected.sort_by(|left, right| left.name().cmp(right.name()));
    assert_eq!(actual, expected);
    assert_eq!(
        after.links.info().max_creation_order,
        before
            .links
            .info()
            .max_creation_order
            .map(|counter| counter + 1)
    );
    assert_eq!(
        test_util_hdf5::dense_links::added_group_link_count(&path, parent),
        0
    );
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
        test_util_hdf5::dense_links::write_inspection_fixture(&path, GroupOrder::Indexed);
    }
    let original = std::fs::read(&path).unwrap();
    let error = group_inspection::inspect_group_links(&path, group_path, source_only)
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
    let file = group_inspection::open_group_inspection(&path, source_only).unwrap();
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
    assert_eq!(file.group_inspection_snapshot().unwrap(), original);
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        u64::try_from(original.len()).unwrap()
    );
    assert_eq!(file.page_publication_snapshot().unwrap(), original);
    drop(file);
    assert_eq!(std::fs::read(&path).unwrap(), original);
    dir.close().unwrap();
}
