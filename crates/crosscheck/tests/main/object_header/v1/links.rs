#![cfg(feature = "__hdf5-1.10")]
//! A group that tracks link creation order, as libhdf5 writes it with the earliest library version
//! as the low bound: a version 1 object header with the group's Link Info and Group Info messages,
//! and its Link messages where the links are compact.

use std::path::Path;

use hdf5::GroupStorageType;
use hdf5::IndexType;
use hdf5::IterationOrder;
use hdf5::plist::group_create::LinkCreationOrder;
use hdf5_pure::AttrValue;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FormatError;
use rstest::rstest;
use test_util::object_header::MessageType;
use test_util::object_header::v1;
use test_util::object_header::v1::Record;
use test_util::widths::Widths;
use test_util_hdf5::__private::c_object;
use test_util_hdf5::file;
use test_util_hdf5::soft_link;
use test_util_hdf5::soft_link::NativeLink;

/// Writes a file whose group `g` tracks link creation order as `order` and holds a dataset `d`, an
/// attribute `units`, the links `soft` and `external`, and `fillers` more soft links.
fn write_link_message_group(path: &Path, order: LinkCreationOrder, fillers: usize) {
    let file = file::libhdf5_create_earliest(path);
    let group = file
        .create_group_builder()
        .with_gcpl(|properties| properties.link_creation_order(order))
        .create("g")
        .unwrap();
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
    group.link_soft("/g/d", "soft").unwrap();
    group
        .link_external("other.h5", "/values", "external")
        .unwrap();
    for index in 0..fillers {
        group.link_soft("d", &format!("filler{index:02}")).unwrap();
    }
    drop(group);
    file.close().unwrap();
}

#[rstest]
#[case::tracked_compact(LinkCreationOrder::Tracked, 0, GroupStorageType::Compact)]
#[case::indexed_dense(LinkCreationOrder::Indexed, 10, GroupStorageType::Dense)]
fn a_sibling_edit_preserves_the_link_messages_of_a_version_1_header(
    #[case] order: LinkCreationOrder,
    #[case] fillers: usize,
    #[case] storage: GroupStorageType,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("link_messages_v1.h5");
    write_link_message_group(&path, order, fillers);
    let (data_address, before) = {
        let reference = file::libhdf5_open(&path);
        let group = reference.group("g").unwrap();
        assert_eq!(group.info().unwrap().storage_type, storage);
        let header = usize::try_from(c_object::header_address(&reference, "g")).unwrap();
        // `v1::chunks` panics unless `g` has a version 1 object header.
        v1::chunks(&std::fs::read(&path).unwrap(), header, Widths::EIGHT);
        (
            c_object::header_address(&reference, "g/d"),
            creation_orders(&group),
        )
    };
    let fillers: Vec<String> = (0..fillers)
        .map(|index| format!("filler{index:02}"))
        .collect();
    let mut expected_orders: Vec<(String, i64)> = ["d", "soft", "external"]
        .into_iter()
        .map(String::from)
        .chain(fillers.iter().cloned())
        .zip(0..)
        .collect();
    assert_eq!(before, expected_orders);

    {
        let session = File::open_rw(&path).unwrap();
        session
            .group("g")
            .unwrap()
            .create_dataset("added", |b| {
                b.with_i32_data(&[3]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let reference = file::libhdf5_open(&path);
    let group = reference.group("g").unwrap();
    let info = group.info().unwrap();
    assert_eq!(info.storage_type, storage);
    let next = i64::try_from(expected_orders.len()).unwrap();
    assert_eq!(info.max_corder, next + 1);
    assert_eq!(group.gcpl().unwrap().link_creation_order(), order);
    expected_orders.push((String::from("added"), next));
    assert_eq!(creation_orders(&group), expected_orders);
    let added_address = c_object::header_address(&reference, "g/added");
    let mut links = soft_link::native_links(&reference, "/g");
    links.sort_by(|(left, _), (right, _)| left.cmp(right));
    let mut expected_links = vec![
        (
            String::from("added"),
            NativeLink::Hard {
                address: added_address,
            },
        ),
        (
            String::from("d"),
            NativeLink::Hard {
                address: data_address,
            },
        ),
        (
            String::from("external"),
            NativeLink::External {
                filename: String::from("other.h5"),
                object_path: String::from("/values"),
            },
        ),
        (
            String::from("soft"),
            NativeLink::Soft {
                target: String::from("/g/d"),
            },
        ),
    ];
    expected_links.extend(fillers.iter().map(|name| {
        (
            name.clone(),
            NativeLink::Soft {
                target: String::from("d"),
            },
        )
    }));
    expected_links.sort_by(|(left, _), (right, _)| left.cmp(right));
    assert_eq!(links, expected_links);
    for name in ["g/d", "g/soft"] {
        assert_eq!(
            reference.dataset(name).unwrap().read_raw::<i32>().unwrap(),
            [1, 2]
        );
    }
    assert_eq!(
        group.attr("units").unwrap().read_scalar::<i32>().unwrap(),
        7
    );
    drop(group);
    drop(reference);

    let file = File::open(&path).unwrap();
    for name in ["g/d", "g/soft"] {
        assert_eq!(file.dataset(name).unwrap().read_i32().unwrap(), [1, 2]);
    }
    assert_eq!(file.dataset("g/added").unwrap().read_i32().unwrap(), [3]);
    assert_eq!(
        file.group("g").unwrap().attrs().unwrap()["units"],
        AttrValue::I32(7)
    );
    let error = file.dataset("g/external").unwrap_err();
    let Error::Format(FormatError::ExternalLinkUnsupported {
        path,
        filename,
        object_path,
    }) = &error
    else {
        panic!("expected ExternalLinkUnsupported, got {error:?}");
    };
    assert_eq!(
        (path.as_str(), filename.as_str(), object_path.as_str()),
        ("g/external", "other.h5", "/values")
    );
}

fn creation_orders(group: &hdf5::Group) -> Vec<(String, i64)> {
    group
        .links(IndexType::CreationOrder, IterationOrder::Increasing)
        .unwrap()
        .into_iter()
        .map(|(name, info)| (name, info.creation_order.unwrap()))
        .collect()
}

#[test]
fn an_edit_of_a_group_mixing_a_symbol_table_with_link_messages_is_rejected_and_leaves_the_file_unchanged()
 {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed_v1.h5");
    {
        let file = file::libhdf5_create_earliest(&path);
        let table = file.create_group("h").unwrap();
        table
            .new_dataset::<i32>()
            .shape([1])
            .create("e")
            .unwrap()
            .write_raw(&[1])
            .unwrap();
        let links = file
            .create_group_builder()
            .with_gcpl(|properties| properties.link_creation_order(LinkCreationOrder::Tracked))
            .create("g")
            .unwrap();
        links.link_soft("/h/e", "soft").unwrap();
        drop((table, links));
        file.close().unwrap();
    }
    let [table, links] = {
        let reference = file::libhdf5_open(&path);
        ["h", "g"].map(|name| usize::try_from(c_object::header_address(&reference, name)).unwrap())
    };
    let mut bytes = std::fs::read(&path).unwrap();
    replace_a_link_message_with_a_symbol_table(&mut bytes, table, links);
    std::fs::write(&path, &bytes).unwrap();

    let error = {
        let session = File::open_rw(&path).unwrap();
        session
            .group("g")
            .unwrap()
            .create_dataset("added", |b| {
                b.with_i32_data(&[3]);
            })
            .unwrap();
        session.commit().unwrap_err()
    };
    assert!(
        matches!(
            error,
            Error::EditUnsupported("a group mixes symbol-table storage with modern link storage")
        ),
        "{error:?}"
    );
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
}

fn replace_a_link_message_with_a_symbol_table(bytes: &mut [u8], table: usize, group: usize) {
    let symbol_table = records(bytes, table)
        .into_iter()
        .find(|record| record.msg_type == MessageType::SYMBOL_TABLE)
        .unwrap();
    let body = bytes[symbol_table.body.clone()].to_vec();
    let link = records(bytes, group)
        .into_iter()
        .find(|record| record.msg_type == MessageType::LINK && record.body.len() >= body.len())
        .unwrap();
    link.set_type(bytes, MessageType::SYMBOL_TABLE);
    test_util::bytes::set_slice_at(bytes, link.body.start, &body);
}

fn records(bytes: &[u8], header: usize) -> Vec<Record> {
    v1::chunks(bytes, header, Widths::EIGHT)
        .into_iter()
        .flat_map(|chunk| chunk.records)
        .collect()
}
