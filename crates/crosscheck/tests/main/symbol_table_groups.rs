#![cfg(feature = "__hdf5-1.10")]
//! Crosschecks symbol-table group paths, dataset values, and serialized link addresses.

use std::path::Path;

use hdf5::plist::file_create::SharedMessageIndex;
use hdf5::plist::file_create::SharedMessageType;
use hdf5::plist::file_create::Sizeof;
use hdf5::plist::file_create::SizeofInfo;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::SuperblockVersion;
use rstest::rstest;
use test_util_hdf5::symbol_table;

use crate::c_object;

#[derive(Clone, Copy, Debug)]
struct Creation {
    userblock: u64,
    sizes: (Sizeof, Sizeof),
    group_leaf_node_k: u32,
    shared_messages: bool,
    superblock_version: SuperblockVersion,
}

const DEFAULT: Creation = Creation {
    userblock: 0,
    sizes: (Sizeof::Bytes8, Sizeof::Bytes8),
    group_leaf_node_k: 0,
    shared_messages: false,
    superblock_version: SuperblockVersion::Zero,
};

const GROUPS: [&str; 2] = ["/", "/nested"];

fn write_libhdf5_file(path: &Path, creation: Creation, links: usize) {
    let Creation {
        userblock,
        sizes: (sizeof_addr, sizeof_size),
        group_leaf_node_k,
        shared_messages,
        superblock_version: _,
    } = creation;
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_earliest())
        .with_fcpl(|p| {
            p.userblock(userblock)
                .sizes(SizeofInfo {
                    sizeof_addr,
                    sizeof_size,
                })
                .sym_k(0, group_leaf_node_k);
            if shared_messages {
                p.shared_mesg_indexes(&[SharedMessageIndex {
                    message_types: SharedMessageType::ATTRIBUTE,
                    min_message_size: 1,
                }]);
            }
            p
        })
        .create(path)
        .unwrap();
    let nested = file.create_group("nested").unwrap();
    for (group, count, value_base) in [(&*file, links, 0i32), (&nested, 3, NESTED_VALUE_BASE)] {
        for index in 0..count {
            group
                .new_dataset::<i32>()
                .shape([1])
                .create(dataset_name(index).as_str())
                .unwrap()
                .write_raw(&[value_base
                    .checked_add(i32::try_from(index).unwrap())
                    .unwrap()])
                .unwrap();
        }
    }
    nested.create_group("leaf").unwrap();
    file.close().unwrap();
}

fn libhdf5_links(path: &Path, group: &str) -> Vec<(String, u64)> {
    let file = hdf5::File::open(path).unwrap();
    let mut links: Vec<(String, u64)> = file
        .group(group)
        .unwrap()
        .member_names()
        .unwrap()
        .into_iter()
        .map(|name| {
            let address =
                c_object::header_address(&file, &format!("{}/{name}", group.trim_end_matches('/')));
            (name, address)
        })
        .collect();
    links.sort();
    links
}

fn dataset_name(index: usize) -> String {
    format!("a dataset whose name fills the local heap {index:03}")
}

fn assert_group_members(
    file: &File,
    path: &str,
    datasets: usize,
    child_group: &str,
    value_base: i32,
) {
    let group = file.group(path).unwrap();
    let expected_names = (0..datasets).map(dataset_name).collect::<Vec<_>>();
    let mut names = group.datasets().unwrap();
    names.sort();
    assert_eq!(names, expected_names);
    assert_eq!(group.groups().unwrap(), [child_group]);
    let child_path = format!("{}/{child_group}", path.trim_end_matches('/'));
    let child = group.group(child_group).unwrap();
    file.group(&child_path).unwrap();
    let error = file.dataset(&child_path).unwrap_err();
    let Error::NotADataset(actual) = error else {
        panic!("expected NotADataset, got {error:?}");
    };
    assert_eq!(actual, child_path.trim_start_matches('/'));
    if child_group == "leaf" {
        assert_eq!(child.groups().unwrap(), Vec::<String>::new());
        assert_eq!(child.datasets().unwrap(), Vec::<String>::new());
    }
    for (index, name) in expected_names.iter().enumerate() {
        let dataset_path = format!("{}/{name}", path.trim_end_matches('/'));
        let expected = [value_base
            .checked_add(i32::try_from(index).unwrap())
            .unwrap()];
        assert_eq!(
            file.dataset(&dataset_path).unwrap().read_i32().unwrap(),
            expected
        );
        assert_eq!(group.dataset(name).unwrap().read_i32().unwrap(), expected);
        let error = file.group(&dataset_path).unwrap_err();
        let Error::NotAGroup(actual) = error else {
            panic!("expected NotAGroup, got {error:?}");
        };
        assert_eq!(actual, dataset_path.trim_start_matches('/'));
    }
}

#[rstest]
#[case::default(DEFAULT, 40)]
#[case::userblock(Creation { userblock: 512, ..DEFAULT }, 40)]
#[case::four_byte_widths(Creation { sizes: (Sizeof::Bytes4, Sizeof::Bytes4), ..DEFAULT }, 40)]
#[case::two_byte_widths(Creation { sizes: (Sizeof::Bytes2, Sizeof::Bytes2), ..DEFAULT }, 20)]
#[case::mixed_widths(Creation { sizes: (Sizeof::Bytes4, Sizeof::Bytes8), ..DEFAULT }, 40)]
#[case::one_link_per_half_node(Creation { group_leaf_node_k: 1, ..DEFAULT }, 40)]
#[case::wide_nodes(Creation { group_leaf_node_k: 16, ..DEFAULT }, 100)]
#[case::superblock_extension_k(
    Creation { group_leaf_node_k: 16, shared_messages: true, superblock_version: SuperblockVersion::Two, ..DEFAULT },
    100
)]
fn both_readers_resolve_symbol_table_links_whose_addresses_match_libhdf5(
    #[case] creation: Creation,
    #[case] links: usize,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("symbol_table.h5");
    write_libhdf5_file(&path, creation, links);

    let expected: Vec<Vec<(String, u64)>> = GROUPS
        .iter()
        .map(|group| libhdf5_links(&path, group))
        .collect();
    assert_eq!(expected[0].len(), links + 1);
    let bytes = std::fs::read(&path).unwrap();
    let file = File::open(&path).unwrap();
    let fixture = symbol_table::Fixture::new(&file, &bytes);
    let root = fixture.links(file.superblock().root_group_address);
    let nested = root.iter().find(|(name, _)| name == "nested").unwrap().1;
    let decoded = [root, fixture.links(nested)].map(|links| {
        links
            .into_iter()
            .map(|(name, address)| (name, address.get()))
            .collect::<Vec<_>>()
    });
    assert_eq!(decoded.as_slice(), expected);
    for file in [file, File::open_streaming(&path).unwrap()] {
        assert_eq!(file.superblock().version, creation.superblock_version);
        assert_eq!(file.superblock().base_address.get(), creation.userblock);
        assert_group_members(&file, "/", links, "nested", 0);
        assert_group_members(&file, "/nested", 3, "leaf", NESTED_VALUE_BASE);
    }
}

const NESTED_VALUE_BASE: i32 = 1000;
