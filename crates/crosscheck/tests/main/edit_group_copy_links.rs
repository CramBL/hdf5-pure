#![cfg(feature = "__hdf5-1.10")]

use std::path::Path;

use hdf5::GroupStorageType;
use hdf5::IndexType;
use hdf5::IterationOrder;
use hdf5::LinkInfo;
use hdf5::LinkType;
use hdf5::plist::group_create::LinkCreationOrder;
use hdf5::plist::link_create::CharEncoding;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FormatError;
use rstest::rstest;
use test_util_hdf5::file;
use test_util_hdf5::object_copy::Origin;
use test_util_hdf5::soft_link;
use test_util_hdf5::soft_link::NativeLink;

#[rstest]
#[case::same_file(Origin::SameFile, 99)]
#[case::other_file(Origin::OtherFile, 7)]
fn a_copied_compact_group_keeps_libhdf5_soft_and_external_links(
    #[case] origin: Origin,
    #[case] absolute_target_value: i32,
) {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.h5");
    write_linked_group(&source);
    let destination = origin.destination(dir.path(), &source);
    let before = LinkState::read(&source, "/g");
    assert_eq!(
        before.creation_order,
        [
            ("d", LinkType::Hard, 0, CharEncoding::Ascii),
            ("σ", LinkType::Hard, 1, CharEncoding::Utf8),
            ("absolute", LinkType::Soft, 2, CharEncoding::Ascii),
            ("relative", LinkType::Soft, 3, CharEncoding::Ascii),
            ("dangling", LinkType::Soft, 4, CharEncoding::Ascii),
            ("external", LinkType::External, 5, CharEncoding::Ascii),
            ("λ", LinkType::Soft, 6, CharEncoding::Utf8),
        ]
        .map(|(link_name, link_type, creation_order, char_encoding)| {
            (
                String::from(link_name),
                LinkInfo {
                    link_type,
                    creation_order: Some(creation_order),
                    char_encoding,
                },
            )
        })
    );

    origin.copy(&source, &destination, "g", "g_copy");

    assert_eq!(LinkState::read(&source, "/g"), before);
    let copied = LinkState::read(&destination, "/g_copy");
    assert_eq!(copied.creation_order, before.creation_order);
    assert_eq!(copied.storage, before.storage);
    assert_eq!(copied.links.len(), before.links.len());
    for ((copied_name, copied_link), (source_name, source_link)) in
        copied.links.iter().zip(&before.links)
    {
        assert_eq!(copied_name, source_name);
        match (copied_link, source_link) {
            (NativeLink::Hard { address: copied }, NativeLink::Hard { address: source }) => {
                assert_ne!(copied, source, "{copied_name:?} targets the source object");
            }
            _ => assert_eq!(copied_link, source_link),
        }
    }

    let values = [
        ("g_copy/d", vec![1, 2]),
        ("g_copy/σ/v", vec![5]),
        ("g_copy/absolute", vec![absolute_target_value]),
        ("g_copy/relative", vec![1, 2]),
        ("g_copy/λ", vec![5]),
    ];
    let reference = file::libhdf5_open(&destination);
    for (path, expected) in &values {
        assert_eq!(
            &reference.dataset(path).unwrap().read_raw::<i32>().unwrap(),
            expected,
            "{path}"
        );
    }
    reference.close().unwrap();

    let file = File::open(&destination).unwrap();
    for (path, expected) in &values {
        assert_eq!(
            &file.dataset(path).unwrap().read_i32().unwrap(),
            expected,
            "{path}"
        );
    }
    let error = file.dataset("g_copy/dangling").unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(
        error,
        FormatError::DanglingSoftLink {
            path: "g_copy/dangling".into(),
            target: "missing".into(),
        }
    );
    let error = file.dataset("g_copy/external").unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(
        error,
        FormatError::ExternalLinkUnsupported {
            path: "g_copy/external".into(),
            filename: "other.h5".into(),
            object_path: "/values".into(),
        }
    );
}

fn write_linked_group(path: &Path) {
    let file = file::libhdf5_create_v18(path);
    write_values(&file, "target", &[99]);
    let group = file
        .create_group_builder()
        .with_gcpl(|properties| properties.link_creation_order(LinkCreationOrder::Tracked))
        .create("g")
        .unwrap();
    write_values(&group, "d", &[1, 2]);
    let subgroup = group
        .create_group_builder()
        .char_encoding(CharEncoding::Utf8)
        .create("σ")
        .unwrap();
    write_values(&subgroup, "v", &[5]);
    group.link_soft("/target", "absolute").unwrap();
    group.link_soft("d", "relative").unwrap();
    group.link_soft("missing", "dangling").unwrap();
    group
        .link_external("other.h5", "/values", "external")
        .unwrap();
    soft_link::link_soft_utf8(&group, "σ/v", "λ");
    assert_eq!(
        group.info().unwrap().storage_type,
        GroupStorageType::Compact
    );
    drop((subgroup, group));
    file.close().unwrap();
}

fn write_values(group: &hdf5::Group, dataset_name: &str, values: &[i32]) {
    group
        .new_dataset::<i32>()
        .shape([values.len()])
        .create(dataset_name)
        .unwrap()
        .write_raw(values)
        .unwrap();
}

#[derive(Debug, PartialEq)]
struct LinkState {
    links: Vec<(String, NativeLink)>,
    creation_order: Vec<(String, LinkInfo)>,
    storage: (GroupStorageType, i64),
}

impl LinkState {
    fn read(path: &Path, group_path: &str) -> Self {
        let reference = file::libhdf5_open(path);
        let group = reference.group(group_path).unwrap();
        let info = group.info().unwrap();
        let state = Self {
            links: soft_link::native_links(&reference, group_path),
            creation_order: group
                .links(IndexType::CreationOrder, IterationOrder::Increasing)
                .unwrap(),
            storage: (info.storage_type, info.max_corder),
        };
        drop(group);
        reference.close().unwrap();
        state
    }
}
