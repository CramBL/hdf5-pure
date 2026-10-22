#![cfg(feature = "__hdf5-1.10")]

use std::path::Path;

use hdf5::plist::file_create::SharedMessageIndex;
use hdf5::plist::file_create::SharedMessageType;
use hdf5::plist::file_create::Sizeof;
use hdf5::plist::file_create::SizeofInfo;
use hdf5_pure::File;
use hdf5_pure::SuperblockVersion;
use rstest::rstest;

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
    for (group, count) in [(&*file, links), (&nested, 3)] {
        for index in 0..count {
            group
                .new_dataset::<i32>()
                .shape([1])
                .create(format!("a dataset whose name fills the local heap {index:03}").as_str())
                .unwrap()
                .write_raw(&[i32::try_from(index).unwrap()])
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

fn hdf5_pure_links(file: &File, group: &str) -> Vec<(String, u64)> {
    let mut links: Vec<(String, u64)> = file
        .group(group)
        .unwrap()
        .__links()
        .unwrap()
        .into_iter()
        .map(|(name, address)| (name, address.get()))
        .collect();
    links.sort();
    links
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
fn both_readers_list_the_links_and_addresses_libhdf5_lists(
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
    for file in [
        File::open(&path).unwrap(),
        File::open_streaming(&path).unwrap(),
    ] {
        assert_eq!(file.superblock().version, creation.superblock_version);
        assert_eq!(file.superblock().base_address.get(), creation.userblock);
        let listed: Vec<Vec<(String, u64)>> = GROUPS
            .iter()
            .map(|group| hdf5_pure_links(&file, group))
            .collect();
        assert_eq!(listed, expected);
    }
}
