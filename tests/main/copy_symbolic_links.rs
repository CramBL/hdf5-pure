use std::collections::BTreeSet;
use std::path::Path;

use hdf5_pure::CharacterSet;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FormatError;
use hdf5_pure::MessageType;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::LinkMessage;
use hdf5_pure_format::__private::LinkTarget;
use hdf5_pure_format::__private::ObjectHeader;
use hdf5_pure_format::__private::OffsetWidth;
use rstest::rstest;
use test_util_hdf5::object_copy::Origin;

#[rstest]
#[case::same_file(Origin::SameFile, 99)]
#[case::other_file(Origin::OtherFile, 7)]
fn a_group_copy_keeps_its_soft_and_external_links(
    #[case] origin: Origin,
    #[case] absolute_target_value: i32,
) {
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir.path().join("source.h5");
    std::fs::write(&source_path, test_util::modern::v2_symbolic_links()).unwrap();
    let destination = origin.destination(dir.path(), &source_path);
    let source_links = group_links(&source_path, "g");
    assert_eq!(
        source_links
            .iter()
            .map(|body| link_fields(body))
            .collect::<Vec<_>>(),
        expected_links()
    );
    let reached = reachable_objects(&destination).len();

    origin.copy(&source_path, &destination, "g", "g_copy");

    assert_eq!(group_links(&source_path, "g"), source_links);
    let copied_links = group_links(&destination, "g_copy");
    assert_eq!(
        copied_links
            .iter()
            .map(|body| link_fields(body))
            .collect::<Vec<_>>(),
        expected_links()
    );
    for (copied, source) in copied_links.iter().zip(&source_links) {
        match (hard_link_target(copied), hard_link_target(source)) {
            (Some(copied_target), Some(source_target)) => {
                assert_ne!(copied_target, source_target);
                let address = copied.len() - ADDRESS_LEN;
                assert_eq!(copied[..address], source[..address]);
            }
            (None, None) => assert_eq!(copied, source),
            targets => panic!("the copy changed a link's kind: {targets:?}"),
        }
    }
    // The copy adds `g_copy`, `d`, `σ`, and `v`, and no object for a soft or an external link.
    assert_eq!(reachable_objects(&destination).len(), reached + 4);

    let file = File::open(&destination).unwrap();
    for (path, expected) in [
        ("g_copy/d", vec![1, 2]),
        ("g_copy/σ/v", vec![5]),
        ("g_copy/absolute", vec![absolute_target_value]),
        ("g_copy/relative", vec![1, 2]),
        ("g_copy/λ", vec![5]),
    ] {
        assert_eq!(
            file.dataset(path).unwrap().read_i32().unwrap(),
            expected,
            "{path}"
        );
    }
    let copy = file.group("g_copy").unwrap();
    assert_eq!(copy.datasets().unwrap(), ["d"]);
    assert_eq!(copy.groups().unwrap(), ["σ"]);
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
    for error in [
        file.dataset("g_copy/external").unwrap_err(),
        copy.group("external").unwrap_err(),
    ] {
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
}

fn expected_links() -> Vec<(String, Option<LinkTarget>, Option<u64>, CharacterSet)> {
    vec![
        ("d".into(), None, Some(0), CharacterSet::Ascii),
        ("σ".into(), None, Some(1), CharacterSet::Utf8),
        (
            "absolute".into(),
            Some(LinkTarget::soft("/target".into())),
            Some(2),
            CharacterSet::Ascii,
        ),
        (
            "relative".into(),
            Some(LinkTarget::soft("d".into())),
            Some(3),
            CharacterSet::Ascii,
        ),
        (
            "dangling".into(),
            Some(LinkTarget::soft("missing".into())),
            Some(4),
            CharacterSet::Ascii,
        ),
        (
            "external".into(),
            Some(LinkTarget::external("other.h5".into(), "/values".into())),
            Some(5),
            CharacterSet::Ascii,
        ),
        (
            "λ".into(),
            Some(LinkTarget::soft("σ/v".into())),
            Some(6),
            CharacterSet::Utf8,
        ),
    ]
}

fn link_fields(body: &[u8]) -> (String, Option<LinkTarget>, Option<u64>, CharacterSet) {
    let LinkMessage {
        name,
        link_target,
        creation_order,
        charset,
    } = LinkMessage::parse_with_offset_width(body, offset_width()).unwrap();
    let symbolic = match link_target {
        LinkTarget::Hard { .. } => None,
        target => Some(target),
    };
    (name, symbolic, creation_order, charset)
}

fn hard_link_target(body: &[u8]) -> Option<u64> {
    match LinkMessage::parse_with_offset_width(body, offset_width())
        .unwrap()
        .link_target
    {
        LinkTarget::Hard {
            object_header_address,
        } => Some(object_header_address.get()),
        LinkTarget::Indirect(_) => None,
    }
}

fn group_links(path: &Path, link_name: &str) -> Vec<Vec<u8>> {
    let image = std::fs::read(path).unwrap();
    let group = link_messages(&image, root_address(path))
        .iter()
        .find(|body| link_fields(body).0 == link_name)
        .and_then(|body| hard_link_target(body))
        .unwrap();
    link_messages(&image, group)
}

fn reachable_objects(path: &Path) -> BTreeSet<u64> {
    let image = std::fs::read(path).unwrap();
    let root = root_address(path);
    let mut reached = BTreeSet::from([root]);
    let mut pending = vec![root];
    while let Some(address) = pending.pop() {
        for target in link_messages(&image, address)
            .iter()
            .filter_map(|body| hard_link_target(body))
        {
            if reached.insert(target) {
                pending.push(target);
            }
        }
    }
    reached
}

fn link_messages(image: &[u8], address: u64) -> Vec<Vec<u8>> {
    ObjectHeader::parse(
        image,
        AccessMode::ReadOnly,
        usize::try_from(address).unwrap(),
        8,
        8,
    )
    .unwrap()
    .messages
    .into_iter()
    .filter(|message| message.msg_type == MessageType::LINK)
    .map(|message| message.data)
    .collect()
}

fn root_address(path: &Path) -> u64 {
    File::open(path)
        .unwrap()
        .superblock()
        .root_group_address
        .get()
}

fn offset_width() -> OffsetWidth {
    OffsetWidth::try_from(8).unwrap()
}

const ADDRESS_LEN: usize = 8;
