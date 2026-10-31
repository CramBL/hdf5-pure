#![cfg(feature = "__hdf5-1.10")]
//! Soft links in a version 1 (symbol-table) group, as libhdf5 writes them with the earliest
//! library version as the low bound. Listing the group returns only its hard links, and an edit
//! that converts the group to the latest format writes each soft link as a Link message.

use std::path::Path;

use hdf5_pure::AttrValue;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FormatError;
use hdf5_pure::SuperblockVersion;
use rstest::rstest;
use test_util::local_heap;
use test_util::symbol_table;
use test_util::symbol_table::Entry;
use test_util::widths::Widths;
use test_util_hdf5::__private::c_object;
use test_util_hdf5::file;
use test_util_hdf5::soft_link;
use test_util_hdf5::soft_link::NativeLink;

/// Writes a file whose group at `group_path` holds a dataset `d`, an attribute `units`, and the
/// soft links `absolute`, `relative`, and `dangling`.
fn write_soft_link_file(path: &Path, group_path: &str) {
    let file = file::libhdf5_create_earliest(path);
    let group = match group_path {
        "/" => file.group("/").unwrap(),
        _ => file.create_group(group_path).unwrap(),
    };
    group
        .new_dataset::<i32>()
        .shape([2])
        .create("d")
        .unwrap()
        .write_raw(&[1, 2])
        .unwrap();
    group
        .new_attr::<i32>()
        .shape(())
        .create("units")
        .unwrap()
        .write_scalar(&7)
        .unwrap();
    group
        .link_soft(&absolute_target(group_path), "absolute")
        .unwrap();
    group.link_soft("d", "relative").unwrap();
    group.link_soft("missing", "dangling").unwrap();
    drop(group);
    file.close().unwrap();
}

fn absolute_target(group_path: &str) -> String {
    format!("{}/d", group_path.trim_end_matches('/'))
}

fn member(group_path: &str, name: &str) -> String {
    format!("{}/{name}", group_path.trim_end_matches('/'))
}

#[test]
fn a_soft_link_does_not_hide_hard_links_in_a_symbol_table_group() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("soft_v1.h5");
    write_soft_link_file(&path, "/");

    let f = File::open(&path).unwrap();
    assert!(
        matches!(
            f.superblock().version(),
            SuperblockVersion::Zero | SuperblockVersion::One
        ),
        "expected a v0/v1 superblock from the earliest libver bound"
    );
    assert_eq!(f.root().datasets().unwrap(), ["d"]);
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap(), [1, 2]);
}

#[rstest]
fn a_sibling_edit_preserves_the_links_and_attributes_of_a_symbol_table_group(
    #[values("/", "/g")] group_path: &str,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("soft_v1_edit.h5");
    write_soft_link_file(&path, group_path);
    let data_address = {
        let reference = file::libhdf5_open(&path);
        c_object::header_address(&reference, &member(group_path, "d"))
    };

    {
        let session = File::open_rw(&path).unwrap();
        session
            .group(group_path)
            .unwrap()
            .create_dataset("added", |b| {
                b.with_i32_data(&[3]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let reference = file::libhdf5_open(&path);
    let added_address = c_object::header_address(&reference, &member(group_path, "added"));
    let soft = |name: &str, target: &str| {
        (
            String::from(name),
            NativeLink::Soft {
                target: String::from(target),
            },
        )
    };
    let hard = |name: &str, address: u64| (String::from(name), NativeLink::Hard { address });
    let mut links = soft_link::native_links(&reference, group_path);
    links.sort_by(|(left, _), (right, _)| left.cmp(right));
    assert_eq!(
        links,
        [
            soft("absolute", &absolute_target(group_path)),
            hard("added", added_address),
            hard("d", data_address),
            soft("dangling", "missing"),
            soft("relative", "d"),
        ]
    );
    for name in ["d", "absolute", "relative"] {
        assert_eq!(
            reference
                .dataset(&member(group_path, name))
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            [1, 2]
        );
    }
    assert_eq!(
        reference
            .group(group_path)
            .unwrap()
            .attr("units")
            .unwrap()
            .read_scalar::<i32>()
            .unwrap(),
        7
    );
    drop(reference);

    let file = File::open(&path).unwrap();
    let group = file.group(group_path).unwrap();
    let mut datasets = group.datasets().unwrap();
    datasets.sort();
    assert_eq!(datasets, ["added", "d"]);
    assert_eq!(group.attrs().unwrap()["units"], AttrValue::I32(7));
    for name in ["d", "absolute", "relative"] {
        assert_eq!(
            file.dataset(&member(group_path, name))
                .unwrap()
                .read_i32()
                .unwrap(),
            [1, 2]
        );
    }
    assert_eq!(
        file.dataset(&member(group_path, "added"))
            .unwrap()
            .read_i32()
            .unwrap(),
        [3]
    );
    let dangling = member(group_path, "dangling");
    let error = file.dataset(&dangling).unwrap_err();
    let Error::Format(FormatError::DanglingSoftLink { path, target }) = &error else {
        panic!("expected DanglingSoftLink, got {error:?}");
    };
    assert_eq!(
        (path.as_str(), target.as_str()),
        (dangling.trim_start_matches('/'), "missing")
    );
}

#[derive(Clone, Copy, Debug)]
enum Damage {
    UnknownCacheType,
    ValueOutsideTheLocalHeap,
}

#[rstest]
fn an_edit_of_a_group_with_a_damaged_symbolic_entry_is_rejected_and_leaves_the_file_unchanged(
    #[values(Damage::UnknownCacheType, Damage::ValueOutsideTheLocalHeap)] damage: Damage,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("damaged_v1_edit.h5");
    write_soft_link_file(&path, "/");
    let mut bytes = std::fs::read(&path).unwrap();
    let expected = damage.apply(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();

    let error = {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("added", |b| {
                b.with_i32_data(&[3]);
            })
            .unwrap();
        session.commit().unwrap_err()
    };
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(error, expected);
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
}

impl Damage {
    /// Damages the symbol table entry of the first soft link in `bytes`, and returns the error an
    /// edit of the group reports.
    fn apply(self, bytes: &mut [u8]) -> FormatError {
        let node = test_util::bytes::sole_signature(bytes, symbol_table::SIGNATURE);
        let symbols = test_util::bytes::u16_at(bytes, node + symbol_table::SYMBOL_COUNT_AT);
        let entry = (0..usize::from(symbols))
            .map(|index| {
                node + symbol_table::NODE_PREFIX + index * Entry::encoded_len(Widths::EIGHT)
            })
            .find(|&entry| {
                test_util::bytes::u32_at(bytes, entry + Entry::cache_type_at(Widths::EIGHT))
                    == symbol_table::SYMBOLIC_LINK
            })
            .unwrap();
        match self {
            Self::UnknownCacheType => {
                test_util::bytes::set_uint_at(
                    bytes,
                    entry + Entry::cache_type_at(Widths::EIGHT),
                    4,
                    3,
                );
                FormatError::InvalidSymbolTableCacheType(3)
            }
            Self::ValueOutsideTheLocalHeap => {
                let heap = test_util::bytes::sole_signature(bytes, local_heap::SIGNATURE);
                let data_segment_size =
                    test_util::bytes::u64_at(bytes, heap + local_heap::DATA_SEGMENT_SIZE_AT);
                test_util::bytes::set_uint_at(
                    bytes,
                    entry + Entry::scratch_pad_at(Widths::EIGHT),
                    4,
                    u32::MAX.into(),
                );
                FormatError::InvalidLocalHeapNameOffset {
                    offset: u32::MAX.into(),
                    data_segment_size,
                }
            }
        }
    }
}
